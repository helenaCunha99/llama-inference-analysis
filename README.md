# Performance Analysis of LLM Inference on CPU with `llama.cpp`

> **Track A1 — Single-Engine Deep Dive**

---

## Authors

- Srey Pheak Sang — `pg55094@alunos.uminho.pt`
- Helena Beatriz Alves Ribeiro da Cunha — `pg61438@alunos.uminho.pt`
- Margarida dos Santos Ferreira — `pg61441@alunos.uminho.pt`

---

## Overview

This project benchmarks Large Language Model (LLM) inference on CPU using the [`llama.cpp`](https://github.com/ggerganov/llama.cpp) framework.

The study evaluates three quantised models across multiple optimisation dimensions.

### Evaluated Models

- **SmolLM2-360M**
- **Qwen2.5-1.5B**
- **Meta-Llama-3.1-8B**

### Optimisation Parameters

| Parameter | Values |
|---|---|
| Quantisation | `Q2_K`, `Q4_K_M`, `Q5_K_M`, `Q8_0` |
| Thread Count | `4`, `16`, `32` |
| Context Length | `128`, `512`, `1024`, `2048` |

### Measured Metrics

- **TTFT** — Time To First Token
- **TPOT** — Time Per Output Token
- **Throughput** — Tokens per second

---

## Repository Structure

```text
.
├── scripts/            # SLURM and Python scripts
├── results/sample/     # Example benchmark CSV results
├── logs/sample/        # Example SLURM logs
├── plots/              # Generated performance figures
└── README.md
```

---

## Scripts Description

| Script | Purpose |
|---|---|
| `compile_llama.sh` | Builds `llama.cpp` on the Deucalion cluster |
| `download_models.sh` | Downloads the base models |
| `download_quant_models.sh` | Downloads quantised GGUF variants |
| `run_benchmark.sh` | Main SLURM benchmarking script |
| `benchmark_client.py` | Python HTTP benchmarking client |
| `prompts.json` | Dataset containing 25 evaluation prompts |
| `extract_metrics.py` | Extracts and aggregates benchmark metrics |
| `plots.py` | Generates performance plots |
| `measure_bandwidth.sh` | Runs STREAM memory bandwidth measurements |

---

## Reproduction Instructions

### 1. Compile `llama.cpp`

```bash
sbatch scripts/compile_llama.sh
```

### 2. Download Models

```bash
sbatch scripts/download_models.sh
sbatch scripts/download_quant_models.sh
```

### 3. Measure Memory Bandwidth

```bash
sbatch scripts/measure_bandwidth.sh
```

### 4. Run Baseline Benchmark

```bash
sbatch --export=MODEL="/path/to/models/SmolLM2-360M-Instruct-Q4_K_M.gguf",N_THREADS=16,N_PARALLEL=1,CTX_SIZE=2048 scripts/run_benchmark.sh
```

### 5. Run Optimisation Sweeps

#### Quantisation Sweep

Vary:

- `Q2_K`
- `Q4_K_M`
- `Q5_K_M`
- `Q8_0`

#### Threading Sweep

Vary:

- `N_THREADS=4`
- `N_THREADS=16`
- `N_THREADS=32`

#### Context Length Sweep

Vary:

- `CTX_SIZE=128`
- `CTX_SIZE=512`
- `CTX_SIZE=1024`
- `CTX_SIZE=2048`

### 6. Extract Results

```bash
python3 scripts/extract_metrics.py
```

### 7. Generate Plots

```bash
python3 scripts/plots.py
```

---

## Key Results

- **Meta-Llama-3.1-8B** is primarily memory-bandwidth bound
- Optimal configuration observed for Llama-8B:
  - `16` threads
  - `Q4_K_M` quantisation
- **SmolLM2** is mostly overhead-dominated
- `Q5_K_M` provides the best performance/quality trade-off:
  - `1.23 ms/token`
  - `122 tok/s`
- Larger context lengths mainly impact **TTFT**, especially under cold-cache conditions

---

## References

- [`llama.cpp`](https://github.com/ggerganov/llama.cpp)
- [STREAM Benchmark](https://www.cs.virginia.edu/stream/)
- Deucalion HPC Cluster Documentation

---

## License

This project was developed for academic purposes at the University of Minho.
