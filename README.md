# Ollama VRAM & Context Benchmark (Boundary Hunter)

A zero-dependency Python benchmarking utility for [Ollama](https://ollama.com) that automatically discovers the **exact maximum GPU layer offload (`num_gpu`)** and **KV-cache context window ceiling (`num_ctx`)** your GPU can sustain before hitting Out-Of-Memory (OOM) errors.

Ideal for tuning local LLMs on fixed-VRAM NVIDIA GPUs (8 GB, 12 GB, 16 GB, 24 GB+) where guessing `num_gpu` or `num_ctx` leads to silent system RAM spillover or CUDA OOM crashes.

## What It Solves

Standard Ollama speed benchmarks only measure tokens per second (`t/s`) at default settings. When running larger models (14B, 27B, 35B, 70B) or long context windows (16k, 32k, 64k tokens), finding the exact hardware stability boundary manually requires tedious trial and error.

This tool automates discovery in two phases:
1. **Phase 1 — GPU Layer Offload Boundary Hunt (`num_gpu`):** Captures a pure-CPU baseline (`num_gpu = 0`), tests full VRAM offload (`num_gpu = 99`), and if an OOM occurs, steps down by 10 layers and fine-tunes upward by 2 layers to find your exact stability ceiling.
2. **Phase 2 — Top-Down Context & KV-Cache Stress Test (`num_ctx`):** Fills the prompt buffer to target context sizes (`32768 -> 16384 -> 8192 -> 4096` tokens) starting at your Phase 1 layer ceiling. If the KV cache triggers an OOM, it steps down `num_gpu` by 2 until the context window runs reliably.

## Key Features

* **Zero External Dependencies:** Uses only the Python 3.11+ standard library (`urllib`, `json`, `csv`, `threading`, `subprocess`, `fnmatch`, `tomllib`). No `pip install` required.
* **Real-Time VRAM & System RAM Telemetry:** Polls NVIDIA GPU VRAM (`nvidia-smi`) and free system RAM every `0.2s` during inference to catch peak memory spikes.
* **Smart Model Filtering & Wildcards:** Target a specific model by index number, exact name, or substring/wildcard pattern (`--model qwen`, `--model "*r1*"`), with an interactive filtered menu when multiple models match.
* **Dual Logging (`session_run.log` + `summary_metrics.csv`):** Prints color-coded progress to the terminal while saving clean, ANSI-stripped transcripts and structured CSV metrics to dated output folders.
* **Headless / Cron Mode (`-q`):** Suppresses terminal output for unattended overnight sweeps.

## Quick Start

1. Ensure Ollama is running (`ollama serve`).
2. Generate a starter configuration file:
   ```bash
   python3 ollama_vram_hunter.py --init-config
   ```
3. *(Optional)* Move the config to `~/.config/ollama-vram-hunter/boundary-hunter.toml` (auto-detected from any working directory) and customize your `out_dir`, `context_tiers`, or step sizes.
4. Run the interactive benchmark menu:
   ```bash
   python3 ollama_vram_hunter.py
   ```

## Usage Examples

* **List all installed models (sorted smallest to largest) and refresh `.model_name` cache:**
  ```bash
  python3 ollama_vram_hunter.py --list-models
  ```
* **Filter models by substring or wildcard (opens a filtered selection menu if multiple match):**
  ```bash
  python3 ollama_vram_hunter.py --model qwen
  ```
* **Run a single model by exact name (layer hunt only, skipping context tests):**
  ```bash
  python3 ollama_vram_hunter.py --model "deepseek-r1:14b" --ctx-mode 0
  ```
* **Unattended / Headless Sweep (quiet mode, logs to `session_run.log` and `summary_metrics.csv`):**
  ```bash
  python3 ollama_vram_hunter.py -q
  ```
* **Use a custom TOML configuration file:**
  ```bash
  python3 ollama_vram_hunter.py --config my-gpu.toml
  ```

## CLI Reference

| Flag | Description |
| :--- | :--- |
| `--list-models` | Print installed models sorted by size, update `.model_name` cache, and exit. |
| `--model TARGET` | Model index (`0` = sweep all), exact model name, or substring/glob pattern (e.g., `qwen`, `"*70b*"`). |
| `--ctx-mode {0,1,2,3}` | `0` = Skip context tests, `1` = Top-down sweep (`32k->4k`), `2` = Single context (`8192`), `3` = Custom. |
| `--custom-ctx N` | Token count to test when `--ctx-mode 3` is selected. |
| `--count N` | Sequence count limit for the generation task (default: `100`). |
| `-q`, `--quiet` | Suppress screen output and write only to `session_run.log`. |
| `--init-config [FILE]` | Generate a starter TOML config file (default: `boundary-hunter.toml`) and exit. |
| `--config FILE` | Path to a custom TOML configuration file. |
| `--host URL` | Ollama API base URL (default: `http://localhost:11434`). |
| `--out-dir DIR` | Output directory for dated benchmark folders (default: `./benchmarks`). |
| `--version` | Show program version and exit. |

## Output Files

Each run creates a dated folder (`YYYY-MM-DD/`) inside your configured output directory containing:
* `summary_metrics.csv` — Timestamp, model name, model type (`Standard` vs. `Reasoning`), phase, context size, `num_gpu` layers, load time, prompt eval time, tokens/sec (`t/s`), min free RAM (GB), peak VRAM (MB), and status (`SUCCESS`, `FAILED_OOM`, `FAILED_EMPTY`).
* `session_run.log` — Clean, full-session console transcript with ANSI color codes stripped.
* `system_hardware_profile.txt` — Snapshot of OS, CPU, and GPU hardware details.
* `<model>_<phase>_ctx<N>_layers<N>.txt` — Raw Ollama JSON API responses for every individual step.

