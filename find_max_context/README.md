# Context Window & Memory Optimization (`find_max_context_launcher.sh`)

This utility automates the search for the maximum sequence length (context window) that each supported Large Language Model (LLM) can process on your available GPU hardware without triggering Out-Of-Memory (OOM) errors.

---

## Overview & Purpose

When extracting computational subgraphs (s-traces) or performing fine-grained model attribution, different architectures and parameter sizes (e.g., Qwen, Mistral, OLMo, Gemma) require vastly different amounts of VRAM per context length token. 

Instead of manually benchmarking context limits for every model individually, `find_max_context_launcher.sh` automates this process by launching background manager tasks (`find_max_context_manager.py`) in parallel across a pre-configured list of models.

---

## Pre-Configured Model Matrix

The launcher script includes pre-configured model identifiers across several popular open-weights model families:

| Model Family | Included Variants |
| :--- | :--- |
| **Mistral** | `mistralai/Mistral-7B-v0.1` |
| **OLMo 2** | `allenai/OLMo-2-0425-1B`, `allenai/OLMo-2-1124-7B`, `allenai/OLMo-2-1124-13B` |
| **Qwen 3** | `Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-4B-Base`, `Qwen/Qwen3-8B-Base` |
| **Gemma 3** | `google/gemma-3-270m` |
| **Qwen 2.5** | `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-14B`, `Qwen/Qwen2.5-32B` |
| **Qwen 2** | `Qwen/Qwen2-0.5B`, `Qwen/Qwen2-1.5B`, `Qwen/Qwen2-7B` |

---

## Usage Guide

### 1. Make the Launcher Executable

Ensure the launcher shell script has execution permissions:

```bash
chmod +x find_max_context_launcher.sh
```

### 2. Launch Context Optimization Managers

Execute the launcher script:

```bash
./find_max_context_launcher.sh
```

Upon execution, the script will:
1. Automatically create the `context_logs/` output directory if it does not already exist.
2. Iterate through the array of target models.
3. Spawn `find_max_context_manager.py` in the background for each model with appropriate logging redirection.

### 3. Monitoring & Management

- **Track Progress Real-Time**:
  ```bash
  tail -f context_logs/*.log
  ```

- **View Specific Model Logs**:
  ```bash
  tail -f context_logs/mistralai_Mistral-7B-v0.1_manager.log
  ```

- **Check Active Processes**:
  ```bash
  ps -ef | grep find_max_context_manager
  ```

- **Terminate Running Managers** (if needed):
  ```bash
  pkill -f find_max_context_manager.py
  ```