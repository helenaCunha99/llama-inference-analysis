#!/bin/bash
# Download the 3 base models

module load Python/3.14.2-GCCcore-15.2.0
pip install --user huggingface_hub

mkdir -p models

# SmolLM2
hf download bartowski/SmolLM2-360M-Instruct-GGUF SmolLM2-360M-Instruct-Q4_K_M.gguf --local-dir models

# Qwen2.5-1.5B
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models

# Llama-3.1-8B (mandatory baseline)
hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models

echo "All models downloaded to models/"
