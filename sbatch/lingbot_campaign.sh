#!/usr/bin/env bash
#SBATCH --job-name=lingbot-campaign
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
pin_gpu_clocks
report_env

"${PYTHON}" -u -m lab.optimize run "${RUN:?RUN must name the Campaign iteration directory}" \
  --until qualify
