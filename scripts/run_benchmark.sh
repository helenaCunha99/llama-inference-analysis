#!/bin/bash
#SBATCH --job-name=llama_bench
#SBATCH --account=f202500010hpcvlabuminhox
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/F202500010HPCVLABUMINHO/uminhocp008/ADED/project2/logs/bench_%j.out
#SBATCH --error=/projects/F202500010HPCVLABUMINHO/uminhocp008/ADED/project2/logs/bench_%j.err

# --- Load modules in correct order ---
module purge
module load GCCcore/14.3.0
module load OpenSSL/3
module load LLVM/20.1.8-GCCcore-14.3.0
module load Python/3.13.5-GCCcore-14.3.0

# Install requests if not available
pip install requests --quiet --user

# --- Paths ---
PROJECT_DIR="/projects/F202500010HPCVLABUMINHO/uminhocp008/ADED/project2"
LLAMA_DIR="$PROJECT_DIR/llama.cpp/build/bin"
MODELS_DIR="$PROJECT_DIR/models"
RESULTS_DIR="$PROJECT_DIR/results"
SCRIPTS_DIR="$PROJECT_DIR/scripts"

# --- Config ---
MODEL="${MODEL:-$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
N_THREADS="${N_THREADS:-16}"
N_PARALLEL="${N_PARALLEL:-1}"
CTX_SIZE="${CTX_SIZE:-2048}"
N_PREDICT="${N_PREDICT:-256}"
HOST="127.0.0.1"
PORT="8080"

mkdir -p "$RESULTS_DIR" "$PROJECT_DIR/logs"

echo "=== llama.cpp Benchmark ==="
echo "Date     : $(date)"
echo "Node     : $(hostname)"
echo "Model    : $MODEL"
echo "Threads  : $N_THREADS"
echo "Parallel : $N_PARALLEL"
echo "Ctx size : $CTX_SIZE"
echo "Job ID   : $SLURM_JOB_ID"

# --- Start server ---
echo ""
echo ">> Starting llama-server..."
"$LLAMA_DIR/llama-server" \
    --model "$MODEL" \
    --threads "$N_THREADS" \
    --parallel "$N_PARALLEL" \
    --ctx-size "$CTX_SIZE" \
    --host "$HOST" \
    --port "$PORT" \
    --log-disable &

SERVER_PID=$!
trap "echo 'Killing server...'; kill $SERVER_PID 2>/dev/null || true" EXIT

# --- Wait for server ---
echo ">> Waiting for server..."
MAX_WAIT=120
ELAPSED=0
until curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Server timeout"; exit 1
    fi
done
echo "   Server ready after ${ELAPSED}s."

# --- Warmup ---
echo ">> Warmup request..."
curl -sf http://$HOST:$PORT/completion \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "n_predict": 16, "stream": false}' > /dev/null

# --- Run benchmark ---
echo ">> Running benchmark..."
MODEL_NAME=$(basename "$MODEL" .gguf)
OUT_FILE="$RESULTS_DIR/results_${MODEL_NAME}_t${N_THREADS}_p${N_PARALLEL}_ctx${CTX_SIZE}_job${SLURM_JOB_ID}.csv"

python3 "$SCRIPTS_DIR/benchmark_client.py" \
    --host "$HOST" \
    --port "$PORT" \
    --prompts "$SCRIPTS_DIR/prompts.json" \
    --n-predict "$N_PREDICT" \
    --trials 3 \
    --output "$OUT_FILE"

echo ""
echo "=== Done! Results saved to: $OUT_FILE ==="
