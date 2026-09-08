#!/bin/bash
#SBATCH --job-name=kernel_trace
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source sbatch/_common.sh
OUT="${REPO_DIR}/artifacts/kernel_trace/${SLURM_JOB_ID}"
mkdir -p "${OUT}"
{
    hostname
    git rev-parse HEAD
    git status --short
    nvcc --version
    nvidia-smi --query-gpu=uuid,name,driver_version,clocks.sm,temperature.gpu --format=csv
} > "${OUT}/environment.txt"
set -x
nvcc -O3 -std=c++17 -lineinfo -arch=sm_90 -Xptxas=-v -Isrc \
    lab/examples/kernel_trace/profile_example.cu -o "${OUT}/profile_example" \
    2> "${OUT}/build.log"
cuobjdump --dump-resource-usage "${OUT}/profile_example" > "${OUT}/resources.txt"
cuobjdump --dump-sass "${OUT}/profile_example" > "${OUT}/sass.txt"
for blocks in 1 64 512; do
    "${OUT}/profile_example" "${OUT}/cta${blocks}.raw.json" "${blocks}" 128
    "${PYTHON}" -m benchmarks.kernel_trace.export_perfetto \
        "${OUT}/cta${blocks}.raw.json" "${OUT}/cta${blocks}.perfetto.json" \
        --summary "${OUT}/cta${blocks}.summary.json"
done
compute-sanitizer --tool memcheck --error-exitcode 1 \
    "${OUT}/profile_example" "${OUT}/memcheck.raw.json" 512 128 \
    > "${OUT}/memcheck.log" 2>&1
