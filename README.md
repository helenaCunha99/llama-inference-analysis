# Performance Analysis of LLM Inference on CPU with llama.cpp

**Track A1 - Single-Engine Deep Dive**
**Big Data Analysis - University of Minho**

## Authors
- Srey Pheak Sang (pg55094@uminho.pt)
- Helena Beatriz Alves Ribeiro da Cunha (pg61438@alunos.uminho.pt)
- Margarida dos Santos Ferreira (pg61441@alunos.uminho.pt)

## Overview
This project benchmarks LLM inference on CPU using llama.cpp. Three quantised models (SmolLM2-360M, Qwen2.5-1.5B, Meta-Llama-3.1-8B) are evaluated across three optimisation dimensions: quantisation levels (Q2_K, Q4_K_M, Q5_K_M, Q8_0), thread counts (4, 16, 32), and context lengths (128, 512, 1024, 2048 tokens). Key metrics: TTFT, TPOT, throughput.

## Repository Structure
- scripts/          # All SLURM and Python scripts
- results/sample/   # Example result CSV
- logs/sample/      # Example SLURM output log
- plots/            # Generated figures (4 plots)

## Scripts Description
- compile_llama.sh           # Build llama.cpp on Deucalion
- download_models.sh         # Download 3 base models
- download_quant_models.sh   # Download quantised variants (Q2_K, Q5_K_M, Q8_0)
- run_benchmark.sh           # Main SLURM benchmark script
- benchmark_client.py        # Python client for HTTP requests
- prompts.json               # Dataset with 25 prompts
- extract_metrics.py         # Post-processing script
- measure_bandwidth.sh       # STREAM bandwidth measurement

## Reproduction Instructions

### 1. Compile llama.cpp
sbatch scripts/compile_llama.sh

### 2. Download Models
sbatch scripts/download_models.sh
sbatch scripts/download_quant_models.sh

### 3. Measure Memory Bandwidth
sbatch scripts/measure_bandwidth.sh

### 4. Run Baseline Benchmark
sbatch --export=MODEL="/path/to/models/SmolLM2-360M-Instruct-Q4_K_M.gguf",N_THREADS=16,N_PARALLEL=1,CTX_SIZE=2048 scripts/run_benchmark.sh

### 5. Run Optimisation Sweeps
- Quantisation: Vary MODEL (Q2_K, Q4_K_M, Q5_K_M, Q8_0)
- Threading: Vary N_THREADS (4, 16, 32)
- Context Length: Vary CTX_SIZE (128, 512, 1024, 2048)

### 6. Extract Results
python3 scripts/extract_metrics.py

## Key Results
- Llama-8B is memory-bandwidth bound (optimal: 16 threads, Q4_K_M)
- SmolLM2 is overhead-dominated
- Quantisation: Q5_K_M offers best trade-off (1.23 ms/token, 122 tok/s)
- Context length primarily affects TTFT on cold caches

## References
- llama.cpp: https://github.com/ggerganov/llama.cpp
- STREAM Benchmark: https://www.cs.virginia.edu/stream/
- Deucalion HPC Cluster Documentation

## License
This project is for academic purposes at the University of Minho.
