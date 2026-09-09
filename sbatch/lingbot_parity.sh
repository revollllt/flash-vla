#!/usr/bin/env bash
#SBATCH --job-name=lingbot-parity
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
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
pin_gpu_clocks
report_env

"${PYTHON}" -u -m eval.lingbot.parity \
  --plan "${PLAN:-shipped}" \
  --seed 42 \
  --layers 36 \
  --steps 10 \
  --out "${OUT:?OUT must name the parity report}"
