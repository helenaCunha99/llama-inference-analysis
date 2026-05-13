#!/bin/bash
# Download additional quantised versions of SmolLM2

module load Python/3.14.2-GCCcore-15.2.0

mkdir -p models

hf download bartowski/SmolLM2-360M-Instruct-GGUF SmolLM2-360M-Instruct-Q2_K.gguf --local-dir models
hf download bartowski/SmolLM2-360M-Instruct-GGUF SmolLM2-360M-Instruct-Q5_K_M.gguf --local-dir models
hf download bartowski/SmolLM2-360M-Instruct-GGUF SmolLM2-360M-Instruct-Q8_0.gguf --local-dir models

echo "Quantised models downloaded to models/"
