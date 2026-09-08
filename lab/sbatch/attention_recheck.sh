#!/bin/bash
#SBATCH --job-name=attn_recheck
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source sbatch/_common.sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
require_cuda
report_env
BASE=artifacts/optimization/async-attention-001
OUT="${BASE}/confirmation_${SLURM_JOB_ID}"
mkdir -p "${OUT}"
compute-sanitizer --tool synccheck --error-exitcode 1 "${PYTHON}" -m lab.optimize.attention_recheck "${BASE}/run_602528" --out "${OUT}/synccheck.json"
compute-sanitizer --tool memcheck --error-exitcode 1 "${PYTHON}" -m lab.optimize.attention_recheck "${BASE}/run_602528" --out "${OUT}/memcheck.json"
"${PYTHON}" -m lab.optimize.attention_recheck "${BASE}/run_602528" --timing --out "${OUT}/unprofiled.json"
