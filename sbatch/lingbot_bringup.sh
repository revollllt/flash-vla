#!/usr/bin/env bash
#SBATCH --job-name=lingbot-bringup
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err

set -euo pipefail

export PYTHON="${LINGBOT_PYTHON:-/data/user/jzou521/codes/cuda/flash-vla/artifacts/envs/lingbot-official-4eb34b7/bin/python}"
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"

LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/compat' | paste -sd: -)"
export LD_LIBRARY_PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}/src:${SLURM_SUBMIT_DIR:-$PWD}"

require_cuda
report_env

OUTPUT="${OUTPUT:-/data/user/jzou521/codes/cuda/flash-vla/artifacts/onboarding/lingbot-vla-4b-h100-bf16/bringup}"
mkdir -p "${OUTPUT}"
"${PYTHON}" -u -m eval.lingbot.parity \
  --plan "${PLAN:-reference}" \
  --seed 42 \
  --out "${OUTPUT}/official-parity-${PLAN:-reference}.json"
