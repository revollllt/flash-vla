#!/usr/bin/env bash
#SBATCH --job-name=lingbot-baselines
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

ROOT="/data/user/jzou521/codes/cuda/flash-vla"
UPSTREAM="${ROOT}/artifacts/upstreams/lingbot-vla"
CHECKPOINT="/data/user/jzou521/models/lingbot-vla-4b-posttrain-robotwin-fb71a2c"
QWEN="${ROOT}/artifacts/upstreams/qwen2.5-vl-3b-instruct"
OUTPUT="${ROOT}/artifacts/onboarding/lingbot-vla-4b-h100-bf16/baselines"
mkdir -p "${OUTPUT}"

for mode in eager compile; do
  "${PYTHON}" -u -m eval.lingbot.reference \
    --upstream "${UPSTREAM}" \
    --checkpoint "${CHECKPOINT}" \
    --qwen "${QWEN}" \
    --output "${OUTPUT}/upstream-${mode}" \
    --mode "${mode}" --seed 42 --layers 36 --steps 10 \
    --soak-s 10 --warmup 5 --reps 100 --timer wall
done

"${PYTHON}" -u -m benchmarks latency \
  --target h100/lingbot_vla \
  --plan reference --plan shipped \
  --seed 42 --layers 36 --steps 10 \
  --warmup 5 --reps 100 --no-attribution \
  --out "${OUTPUT}/flash-reference-shipped.json"
