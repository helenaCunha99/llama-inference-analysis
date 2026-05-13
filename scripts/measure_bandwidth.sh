#!/bin/bash
#SBATCH --account=f202500010hpcvlabuminhox
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --job-name=stream_bw
#SBATCH --output=stream_output.txt
#SBATCH --error=stream_error.txt

cd /tmp
wget -O stream.c https://www.cs.virginia.edu/stream/FTP/Code/stream.c
gcc -O3 -fopenmp -march=native -DSTREAM_ARRAY_SIZE=100000000 stream.c -o stream

for T in 1 16 32 64 128; do
    echo "=== OMP_NUM_THREADS=$T ==="
    OMP_NUM_THREADS=$T ./stream | grep Triad
done
