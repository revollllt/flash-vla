#!/usr/bin/env bash
#SBATCH --job-name=lingbot-correctness
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

ROOT="/data/user/jzou521/codes/cuda/flash-vla"
UPSTREAM="${ROOT}/artifacts/upstreams/lingbot-vla"
CHECKPOINT="/data/user/jzou521/models/lingbot-vla-4b-posttrain-robotwin-fb71a2c"
QWEN="${ROOT}/artifacts/upstreams/qwen2.5-vl-3b-instruct"
OUTPUT="${ROOT}/artifacts/onboarding/lingbot-vla-4b-h100-bf16/correctness"
mkdir -p "${OUTPUT}"

"${PYTHON}" -u -m eval.smoke --json > "${OUTPUT}/declaration-smoke.json"

for case in shallow full-single-step; do
  if [[ "${case}" == "shallow" ]]; then
    layers=1
  else
    layers=36
  fi
  oracle="${OUTPUT}/${case}"
  "${PYTHON}" -u -m eval.lingbot.reference \
    --upstream "${UPSTREAM}" \
    --checkpoint "${CHECKPOINT}" \
    --qwen "${QWEN}" \
    --output "${oracle}" \
    --mode eager --seed 42 --layers "${layers}" --steps 1 --warmup 0 --reps 1
  "${PYTHON}" -u -m eval.lingbot.parity \
    --plan reference --oracle "${oracle}" --seed 42 \
    --layers "${layers}" --steps 1 \
    --out "${OUTPUT}/${case}-parity.json"
done
