#!/bin/bash
# Compilation script for llama.cpp on Deucalion

module load gcc/13.3.0
module load CMake/3.31.8-GCCcore-13.3.0

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DLLAMA_BUILD_SERVER=ON
make -j4

echo "Compilation complete. Binaries in llama.cpp/build/bin/"
