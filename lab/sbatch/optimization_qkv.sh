#!/bin/bash
#SBATCH --job-name=optimization_qkv
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:12:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source sbatch/_common.sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
require_cuda
report_env
"${PYTHON}" -u -m lab.optimize.qkv_probe --inputs artifacts/optimization/qkv-inputs --out "artifacts/optimization/qkv_${SLURM_JOB_ID}"
