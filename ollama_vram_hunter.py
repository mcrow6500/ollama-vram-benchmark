#!/usr/bin/env python3
"""
Ollama VRAM Boundary Hunter & Context Stress Tester
Finds the maximum stable GPU layer offload and KV-cache context ceiling
for local Ollama models before hitting Out-Of-Memory (OOM) errors.
"""

import argparse
import csv
import datetime
import fnmatch
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import tomllib
except ImportError:
    tomllib = None

__version__ = "1.1.0"

DEFAULT_CONFIG_FILENAME = "boundary-hunter.toml"
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

DEFAULT_CONFIG = {
    "general": {
        "config_file": DEFAULT_CONFIG_FILENAME,
        "host": "http://localhost:11434",
        "out_dir": "./benchmarks",
        "default_count_limit": 100,
    },
    "benchmark": {
        "context_tiers": list((32768, 16384, 8192, 4096)),
        "single_context_size": 8192,
        "base_context_size": 2048,
        "max_layers": 99,
        "coarse_step": 10,
        "fine_step": 2,
    },
    "classification": {
        "reasoning_keywords": ["nail", "r1", "reasoning", "qwq"],
    },
}

STARTER_TOML_TEMPLATE = """# Ollama VRAM Boundary Hunter Configuration

[general]
# Default config filename (can point to an alternate .toml file)
config_file = "{config_file}"
host = "http://localhost:11434"
out_dir = "./benchmarks"
default_count_limit = 100

[benchmark]
# Context sizes tested top-down in Phase 2 (Mode 1)
context_tiers = [32768, 16384, 8192, 4096]
single_context_size = 8192
base_context_size = 2048

# Layer search boundaries and step sizes
max_layers = 99
coarse_step = 10
fine_step = 2

[classification]
# Substrings (case-insensitive) used to tag models as "Reasoning" in the CSV
reasoning_keywords = ["nail", "r1", "reasoning", "qwq"]
"""


def deep_merge(base, override):
    """Recursively merges user TOML settings over built-in defaults."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_toml_file(path_obj):
    if not tomllib or not path_obj.is_file():
        return {}
    with open(path_obj, "rb") as f:
        return tomllib.load(f)


def resolve_configuration(cli_config_path=None):
    """
    Loads configuration with precedence:
    CLI --config > Custom pointer in default TOML > Default TOML > Built-in defaults.
    """
    config = deep_merge(DEFAULT_CONFIG, {})
    active_path = None

    if cli_config_path:
        target = Path(cli_config_path).expanduser()
        if not target.is_file():
            print(f"Error: Specified config file not found: {target}")
            sys.exit(1)
        data = load_toml_file(target)
        config = deep_merge(config, data)
        return config, target

    search_candidates = [
        Path.cwd() / DEFAULT_CONFIG_FILENAME,
        Path.home() / ".config" / "ollama-vram-hunter" / DEFAULT_CONFIG_FILENAME,
    ]

    for candidate in search_candidates:
        if candidate.is_file():
            data = load_toml_file(candidate)
            config = deep_merge(config, data)
            active_path = candidate

            custom_name = config["general"].get("config_file", DEFAULT_CONFIG_FILENAME)
            if custom_name and custom_name != candidate.name:
                redirect_path = Path(custom_name).expanduser()
                if not redirect_path.is_absolute():
                    redirect_path = candidate.parent / redirect_path
                if redirect_path.is_file():
                    redirect_data = load_toml_file(redirect_path)
                    config = deep_merge(config, redirect_data)
                    active_path = redirect_path
            break

    return config, active_path
class DualLogger:
    """Writes colored output to stdout (unless quiet) and clean text to a log file."""

    def __init__(self, log_path, quiet=False):
        self.log_path = Path(log_path)
        self.quiet = quiet

    def log(self, message="", end="\n"):
        if not self.quiet:
            print(message, end=end, flush=True)
        clean_text = ANSI_ESCAPE_RE.sub("", message)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(clean_text + end)


class ResourceMonitor:
    """Polls System RAM and NVIDIA VRAM in a background thread during inference."""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.min_free_ram_kb = float("inf")
        self.peak_vram_mb = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _get_free_ram_kb(self):
        if os.path.exists("/proc/meminfo"):
            try:
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            return int(line.split()[1])
            except Exception:
                pass

        sys_os = platform.system()
        try:
            if sys_os == "Windows":
                out = subprocess.check_output(
                    ["wmic", "OS", "get", "FreePhysicalMemory", "/Value"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                for line in out.splitlines():
                    if "FreePhysicalMemory=" in line:
                        return int(line.split("=")[1].strip())
            elif sys_os == "Darwin":
                out = subprocess.check_output(
                    ["vm_stat"], text=True, stderr=subprocess.DEVNULL
                )
                free_pages = 0
                for line in out.splitlines():
                    if "Pages free:" in line or "Pages inactive:" in line:
                        free_pages += int(re.sub(r"\D", "", line))
                return (free_pages * 4096) // 1024
        except Exception:
            pass
        return None

    def _get_used_vram_mb(self):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            vals = [int(x.strip()) for x in out.strip().splitlines() if x.strip().isdigit()]
            return sum(vals) if vals else 0
        except Exception:
            return 0

    def _poll(self):
        while not self._stop_event.is_set():
            ram_kb = self._get_free_ram_kb()
            if ram_kb is not None and ram_kb < self.min_free_ram_kb:
                self.min_free_ram_kb = ram_kb

            vram_mb = self._get_used_vram_mb()
            if vram_mb > self.peak_vram_mb:
                self.peak_vram_mb = vram_mb

            time.sleep(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        free_gb = (
            round(self.min_free_ram_kb / 1048576, 2)
            if self.min_free_ram_kb != float("inf")
            else 0.0
        )
        return free_gb, self.peak_vram_mb
class OllamaVramHunter:
    def __init__(self, config, host_override=None, out_dir_override=None, count_override=None, quiet=False):
        self.cfg = config
        self.host = (host_override or self.cfg["general"]["host"]).rstrip("/")
        out_dir = out_dir_override or self.cfg["general"]["out_dir"]
        self.count_limit = count_override or self.cfg["general"]["default_count_limit"]

        today_str = datetime.date.today().strftime("%Y-%m-%d")
        self.todays_dir = Path(out_dir).expanduser() / today_str
        self.todays_dir.mkdir(parents=True, exist_ok=True)

        self.sys_info_file = self.todays_dir / "system_hardware_profile.txt"
        self.csv_out_file = self.todays_dir / "summary_metrics.csv"
        self.session_log_file = self.todays_dir / "session_run.log"
        self.logger = DualLogger(self.session_log_file, quiet=quiet)

        self._init_csv()

    def _init_csv(self):
        if not self.csv_out_file.exists():
            headers = [
                "Timestamp",
                "Model Name",
                "Model Type",
                "Test Phase",
                "Context Size",
                "Count Limit",
                "GPU Layers",
                "Load Time (sec)",
                "Prompt Eval Time (sec)",
                "Tokens Generated",
                "Gen Duration (sec)",
                "Tokens Per Sec",
                "Min Free RAM (GB)",
                "Peak VRAM (MB)",
                "Status",
            ]
            with open(self.csv_out_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def write_hardware_profile(self):
        if self.sys_info_file.exists():
            return
        lines = [
            "--- System Hardware Profile ---",
            f"OS: {platform.platform()}",
            f"Processor: {platform.processor()}",
        ]
        try:
            gpu = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=gpu_name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            lines.append(f"GPU: {gpu}")
        except Exception:
            lines.append("GPU: nvidia-smi unavailable")

        self.sys_info_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def get_installed_models(self, write_cache=True):
        req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            self.logger.log(f"Error: Could not connect to Ollama at {self.host} ({exc})")
            sys.exit(1)

        models = data.get("models", [])
        if not models:
            self.logger.log("Error: No models found installed in Ollama.")
            sys.exit(1)

        models.sort(key=lambda m: m.get("size", 0))
        parsed = []
        for m in models:
            size_gb = m.get("size", 0) / (1024**3)
            parsed.append((m["name"], f"{size_gb:.1f} GB"))

        if write_cache:
            cache_dir = Path.home() / ".config" / "ollama-vram-hunter"
            cache_file = (cache_dir / ".model_name") if cache_dir.is_dir() else (Path.cwd() / ".model_name")
            lines = [f"[{idx}] {name} ({size})" for idx, (name, size) in enumerate(parsed, start=1)]
            try:
                cache_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            except Exception:
                pass

        return parsed

    def unload_model(self, model_name):
        payload = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15):
                pass
        except Exception:
            pass

    def _build_prompt(self, test_phase, ctx_size):
        base_task = (
            f"List the numbers from 1 to {self.count_limit} sequentially in digits, "
            "separated by a single space. Output absolutely nothing else."
        )
        if test_phase != "ContextTest":
            return base_task

        fill_words = (ctx_size * 3) // 4
        repeats = max(1, fill_words // 5)
        filler = "benchmark memory allocation test token " * repeats
        return f"{filler}Now ignore the filler text above. {base_task}"

    def _log_csv(self, row):
        with open(self.csv_out_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def run_benchmark(self, model_name, model_type, target_layers, test_phase, ctx_size, is_base_run):
        safe_name = re.sub(r"[/:]", "-", model_name)
        out_file = self.todays_dir / f"{safe_name}_{test_phase}_ctx{ctx_size}_layers{target_layers}.txt"

        prompt = self._build_prompt(test_phase, ctx_size)
        payload = json.dumps(
            {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "-1m",
                "options": {
                    "num_gpu": target_layers,
                    "num_ctx": ctx_size,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "seed": 42,
                },
            }
        ).encode("utf-8")

        monitor = ResourceMonitor(interval=0.2)
        monitor.start()

        response_data = {}
        api_error = ""
        try:
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3600) as resp:
                raw_text = resp.read().decode("utf-8")
                response_data = json.loads(raw_text)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            api_error = err_body or str(e)
            response_data = {"error": api_error}
        except Exception as e:
            api_error = str(e)
            response_data = {"error": api_error}

        min_ram_gb, peak_vram_mb = monitor.stop()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header = (
            f"--- BENCHMARK RUN: {timestamp} ---\n"
            f"Model: {model_name} | Phase: {test_phase} | Context: {ctx_size} | GPU Layers: {target_layers}\n"
        )
        out_file.write_text(header + json.dumps(response_data, indent=2) + "\n", encoding="utf-8")

        if api_error or "error" in response_data or not response_data:
            self.logger.log(
                f"  ↳ [{test_phase} | Ctx: {ctx_size}] Layers: {target_layers:2d} | "
                f"Min Free RAM: {min_ram_gb:5.2f} GB | Peak VRAM: {peak_vram_mb:5d} MB | "
                f"Status: \033[31mFAILED_OOM\033[0m"
            )
            if is_base_run:
                self._log_csv([
                    timestamp, model_name, model_type, test_phase, ctx_size,
                    self.count_limit, target_layers, 0, 0, 0, 0, 0,
                    min_ram_gb, peak_vram_mb, "FAILED_OOM"
                ])
            return False

        load_sec = response_data.get("load_duration", 0) / 1e9
        peval_sec = response_data.get("prompt_eval_duration", 0) / 1e9
        eval_cnt = response_data.get("eval_count", 0)
        eval_sec = response_data.get("eval_duration", 0) / 1e9

        if eval_cnt == 0 or eval_sec <= 0:
            self.logger.log(
                f"  ↳ [{test_phase} | Ctx: {ctx_size}] Layers: {target_layers:2d} | "
                f"Min Free RAM: {min_ram_gb:5.2f} GB | Peak VRAM: {peak_vram_mb:5d} MB | "
                f"Status: \033[33mFAILED_EMPTY\033[0m"
            )
            if is_base_run:
                self._log_csv([
                    timestamp, model_name, model_type, test_phase, ctx_size,
                    self.count_limit, target_layers, round(load_sec, 3), round(peval_sec, 3),
                    0, 0, 0, min_ram_gb, peak_vram_mb, "FAILED_EMPTY"
                ])
            return False

        tps = eval_cnt / eval_sec
        self.logger.log(
            f"  ↳ [{test_phase} | Ctx: {ctx_size}] Layers: {target_layers:2d} | "
            f"Speed: {tps:6.2f} t/s | Load: {load_sec:5.2f}s | "
            f"Min Free RAM: {min_ram_gb:5.2f} GB | Peak VRAM: {peak_vram_mb:5d} MB | "
            f"Status: \033[32mSUCCESS\033[0m"
        )
        self._log_csv([
            timestamp, model_name, model_type, test_phase, ctx_size,
            self.count_limit, target_layers, round(load_sec, 3), round(peval_sec, 3),
            eval_cnt, round(eval_sec, 3), round(tps, 2), min_ram_gb, peak_vram_mb, "SUCCESS"
        ])
        return True
    def hunt_model(self, model_name, ctx_tiers):
        bench_cfg = self.cfg["benchmark"]
        base_ctx = bench_cfg["base_context_size"]
        max_layers = bench_cfg["max_layers"]
        coarse_step = bench_cfg["coarse_step"]
        fine_step = bench_cfg["fine_step"]

        keywords = [k.lower() for k in self.cfg["classification"]["reasoning_keywords"]]
        lower = model_name.lower()
        model_type = "Reasoning" if any(k in lower for k in keywords) else "Standard"

        self.logger.log("\n" + "-" * 60)
        self.logger.log(f" 🎯 PHASE 1: VRAM Layer Boundary Hunter for {model_name}")
        self.logger.log("-" * 60)

        self.logger.log(f"⏳ Step 1: Capturing baseline CPU performance ({base_ctx} ctx)...")
        self.run_benchmark(model_name, model_type, 0, "LayerHunt", base_ctx, True)
        max_stable_layers = 0

        self.logger.log(f"⏳ Step 2: Testing maximum offload ({max_layers} layers)...")
        if self.run_benchmark(model_name, model_type, max_layers, "LayerHunt", base_ctx, False):
            self.logger.log(f"🎉 Model fits 100% into VRAM at {max_layers} layers.")
            max_stable_layers = max_layers
        else:
            self.logger.log("💥 Max layers triggered an OOM error. Commencing descending search...")
            low_bound = 0
            start_coarse = max_layers - (max_layers % coarse_step)
            if start_coarse == max_layers:
                start_coarse -= coarse_step

            for layer_try in range(start_coarse, 0, -coarse_step):
                self.logger.log(f"📉 Testing {layer_try} layers...")
                if self.run_benchmark(model_name, model_type, layer_try, "LayerHunt", base_ctx, False):
                    low_bound = layer_try
                    break

            self.logger.log(f"🔬 Stable foundation found at {low_bound} layers. Narrowing down boundary...")
            for up_step in range(low_bound + fine_step, min(low_bound + coarse_step, max_layers), fine_step):
                self.logger.log(f"🔬 Fine-tuning: Testing {up_step} layers...")
                if not self.run_benchmark(model_name, model_type, up_step, "LayerHunt", base_ctx, False):
                    self.logger.log("🛑 VRAM Boundary Breach detected!")
                    break
                low_bound = up_step

            max_stable_layers = low_bound

        self.logger.log("-" * 60)
        self.logger.log(f" 📊 Phase 1 Max Stable Layers ({base_ctx} ctx): {max_stable_layers}")
        self.logger.log("-" * 60)

        if ctx_tiers and max_stable_layers > 0:
            for target_ctx in ctx_tiers:
                self.logger.log("\n" + "-" * 60)
                self.logger.log(f" 🧠 PHASE 2: Context Memory Test ({target_ctx} tokens) for {model_name}")
                self.logger.log(f" Starting at Phase 1 ceiling: {max_stable_layers} layers")
                self.logger.log("-" * 60)

                self.unload_model(model_name)
                ctx_stable_layers = max_stable_layers

                if self.run_benchmark(model_name, model_type, max_stable_layers, "ContextTest", target_ctx, True):
                    self.logger.log(f"🎉 Model held {target_ctx} context at full {max_stable_layers} layers!")
                else:
                    self.logger.log(
                        f"💥 Context KV cache triggered OOM at {max_stable_layers} layers. "
                        f"Stepping down by {fine_step}..."
                    )
                    ctx_stable_layers = 0
                    for test_layer in range(max_stable_layers - fine_step, 0, -fine_step):
                        self.logger.log(f"📉 Testing Context ({target_ctx}) at {test_layer} layers...")
                        if self.run_benchmark(model_name, model_type, test_layer, "ContextTest", target_ctx, False):
                            ctx_stable_layers = test_layer
                            break

                self.logger.log("-" * 60)
                self.logger.log(f" 📊 Max Layers @ {target_ctx} Ctx: {ctx_stable_layers}")
                self.logger.log("-" * 60)

        self.unload_model(model_name)


def main():
    parser = argparse.ArgumentParser(description="Ollama VRAM Boundary Hunter & Context Stress Tester")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", default=None, help=f"Path to TOML config file (default: {DEFAULT_CONFIG_FILENAME})")
    parser.add_argument(
        "--init-config",
        nargs="?",
        const=DEFAULT_CONFIG_FILENAME,
        default=None,
        help=f"Generate a starter config file (default: {DEFAULT_CONFIG_FILENAME}) and exit",
    )
    parser.add_argument("--host", default=None, help="Ollama API base URL (overrides config)")
    parser.add_argument("--out-dir", default=None, help="Output directory for logs and CSV (overrides config)")
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress screen output and write only to session_run.log",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List installed models (sorted smallest to largest), refresh .model_name, and exit",
    )
    parser.add_argument("--model", type=str, default=None, help="Model index (0 = sweep all models)")
    parser.add_argument("--count", type=int, default=None, help="Number sequence count limit (overrides config)")
    parser.add_argument(
        "--ctx-mode",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        help="Context mode: 0=Skip, 1=Top-down sweep, 2=Single context size, 3=Custom",
    )
    parser.add_argument("--custom-ctx", type=int, default=None, help="Custom context size when --ctx-mode 3 is used")
    args = parser.parse_args()

    if args.init_config is not None:
        target_path = Path(args.init_config).expanduser()
        if target_path.exists():
            print(f"Error: {target_path} already exists. Remove it or specify a different filename.")
            sys.exit(1)
        target_path.write_text(
            STARTER_TOML_TEMPLATE.format(config_file=target_path.name),
            encoding="utf-8",
        )
        print(f"Starter configuration written to: {target_path.resolve()}")
        sys.exit(0)

    if tomllib is None and args.config is not None:
        print("Warning: Python 3.11+ is required to parse TOML config files. Using built-in defaults.")

    config, loaded_cfg_path = resolve_configuration(args.config)
    bench_cfg = config["benchmark"]

    hunter = OllamaVramHunter(
        config=config,
        host_override=args.host,
        out_dir_override=args.out_dir,
        count_override=args.count,
        quiet=args.quiet,
    )

    hunter.logger.log(f"\n=== Session Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    if loaded_cfg_path:
        hunter.logger.log(f"Loaded config: {loaded_cfg_path}")
    else:
        hunter.logger.log("Loaded config: Built-in defaults (no TOML file found)")
    hunter.logger.log("--- Querying Ollama Environment ---")

    models = hunter.get_installed_models(write_cache=True)

    if args.list_models:
        print("Installed Ollama Models (Sorted Smallest -> Largest):")
        print("  [0] Sweep ALL models")
        for idx, (m_name, m_size) in enumerate(models, start=1):
            print(f"  [{idx}] {m_name} ({m_size})")
        sys.exit(0)

    def prompt_model_menu(candidate_list, header_text):
        print(header_text)
        print(f"  [0] Sweep ALL {len(candidate_list)} models (Smallest to Largest)")
        for idx, (m_name, m_size) in enumerate(candidate_list, start=1):
            print(f"  [{idx}] {m_name} ({m_size})")
        while True:
            choice = input(f"Enter number (0-{len(candidate_list)}): ").strip()
            if choice.isdigit() and 0 <= int(choice) <= len(candidate_list):
                pick = int(choice)
                return candidate_list if pick == 0 else [candidate_list[pick - 1]]
            print("Invalid selection.")

    # Resolve target_models as a list of (name, size) tuples
    target_models = []
    if args.model is not None:
        sel = args.model.strip()
        if sel.isdigit():
            idx = int(sel)
            if idx == 0:
                target_models = models
            elif 1 <= idx <= len(models):
                target_models = [models[idx - 1]]
            else:
                hunter.logger.log(f"Error: --model index must be between 0 and {len(models)}.")
                sys.exit(1)
        else:
            # Support both explicit globs (*qwen*) and plain substring matching (Kai)
            pattern = sel.lower() if any(ch in sel for ch in "*?[]") else f"*{sel.lower()}*"
            matches = [
                (m_name, m_size)
                for m_name, m_size in models
                if fnmatch.fnmatch(m_name.lower(), pattern)
            ]
            if not matches:
                hunter.logger.log(f"Error: No installed models matched '{sel}'.")
                sys.exit(1)
            elif len(matches) == 1 or args.quiet:
                target_models = matches
            else:
                target_models = prompt_model_menu(
                    matches,
                    f"\nMultiple models matched '{sel}' (Sorted Smallest -> Largest):",
                )
    else:
        if args.quiet:
            target_models = models
        else:
            target_models = prompt_model_menu(
                models,
                "Select an option for VRAM Boundary Hunting (Sorted Smallest -> Largest):",
            )

    if args.count is None and not args.quiet:
        default_cnt = hunter.count_limit
        count_in = input(f"Enter count limit [Default: {default_cnt}]: ").strip()
        if count_in.isdigit() and int(count_in) > 0:
            hunter.count_limit = int(count_in)

    ctx_mode = args.ctx_mode
    default_tiers = bench_cfg["context_tiers"]
    single_ctx = bench_cfg["single_context_size"]
    base_ctx = bench_cfg["base_context_size"]

    if ctx_mode is None:
        if args.quiet:
            ctx_mode = 1
        else:
            print("\nSelect Context Memory Stress Test Mode:")
            print(f"  [0] Skip Context Tests (Layer hunt at {base_ctx} ctx only)")
            print(f"  [1] Top-Down Context Sweep ({' '.join(map(str, default_tiers))} tokens) [Default]")
            print(f"  [2] Single Context Test ({single_ctx} tokens)")
            print("  [3] Custom Context Size")
            while True:
                raw_mode = input("Enter context mode (0-3) [Default: 1]: ").strip() or "1"
                if raw_mode in ("0", "1", "2", "3"):
                    ctx_mode = int(raw_mode)
                    break
                print("Invalid selection.")

    if ctx_mode == 0:
        ctx_tiers = []
    elif ctx_mode == 1:
        ctx_tiers = default_tiers
    elif ctx_mode == 2:
        ctx_tiers = [single_ctx]
    else:
        if args.custom_ctx and args.custom_ctx > 0:
            ctx_tiers = [args.custom_ctx]
        elif not args.quiet:
            while True:
                custom = input("Enter custom context size in tokens (e.g., 65536): ").strip()
                if custom.isdigit() and int(custom) > 0:
                    ctx_tiers = [int(custom)]
                    break
                print("Invalid token count.")
        else:
            hunter.logger.log("Error: --custom-ctx is required when using --ctx-mode 3 in quiet mode.")
            sys.exit(1)

    hunter.write_hardware_profile()

    if len(target_models) > 1:
        hunter.logger.log(f"\n🚀 Starting Sweep across {len(target_models)} matching models (Smallest -> Largest)...")
        for i, (m_name, m_size) in enumerate(target_models, start=1):
            hunter.logger.log(f"\n>>> [Model {i} of {len(target_models)}]: {m_name} ({m_size}) <<<")
            hunter.hunt_model(m_name, ctx_tiers)
    else:
        chosen_name, chosen_size = target_models[0]
        hunter.logger.log(f"\n>>> [Single Model]: {chosen_name} ({chosen_size}) <<<")
        hunter.hunt_model(chosen_name, ctx_tiers)

    hunter.logger.log("\n" + "-" * 60)
    hunter.logger.log(" ✅ BENCHMARK COMPLETE")
    hunter.logger.log(f" Session transcript saved at:   {hunter.session_log_file}")
    hunter.logger.log(f" Final metrics sheet compiled at: {hunter.csv_out_file}")
    hunter.logger.log("-" * 60)


if __name__ == "__main__":
    main()  

