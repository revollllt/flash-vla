#!/bin/bash
#SBATCH --job-name=trace_cal
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source sbatch/_common.sh
OUT="artifacts/kernel_trace/calibration_${SLURM_JOB_ID}"
mkdir -p "${OUT}"
report_env > "${OUT}/environment.txt"
nvcc --version >> "${OUT}/environment.txt"
nvcc -O3 -std=c++17 -lineinfo -arch=sm_90 -Xptxas=-v -Isrc \
    lab/examples/kernel_trace/profile_example.cu -o "${OUT}/profile_example" 2> "${OUT}/build.log"
"${OUT}/profile_example" --calibrate "${OUT}/calibration.json"
"${OUT}/profile_example" "${OUT}/coarse.raw.json" 512 128
"${PYTHON}" -m benchmarks.kernel_trace.export_perfetto "${OUT}/coarse.raw.json" \
    "${OUT}/coarse.perfetto.json" --summary "${OUT}/coarse.summary.json"
cuobjdump --dump-resource-usage "${OUT}/profile_example" > "${OUT}/resources.txt"
cuobjdump --dump-sass "${OUT}/profile_example" > "${OUT}/sass.txt"
