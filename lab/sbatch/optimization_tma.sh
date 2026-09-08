#!/bin/bash
#SBATCH --job-name=optimization_tma
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:12:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source sbatch/_common.sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUTLASS_DIR="${REPO_DIR}/third_party/cutlass"
require_cuda
report_env
OUT="artifacts/optimization/tma_${SLURM_JOB_ID}"
mkdir -p "${OUT}"
export HW_UNIT_TEST_CACHE="${REPO_DIR}/${OUT}/build"
"${PYTHON}" -u .claude/skills/hardware-unit-test/probes/units/tma_ring/tma_ring.py --sweeps A --regime l2 --reps 30 --json "${OUT}/l2.json"
