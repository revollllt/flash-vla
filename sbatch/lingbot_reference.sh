#!/usr/bin/env bash
#SBATCH --job-name=lingbot-reference
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err

set -euo pipefail

export PYTHON="${LINGBOT_PYTHON:-/data/user/jzou521/codes/cuda/flash-vla/artifacts/envs/lingbot-official-4eb34b7/bin/python}"
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"

# The official environment uses a CUDA 12.x torch build, which runs natively
# on both driver generations in this partition and must not inherit CUDA 13's
# forward-compat library from the project environment.
LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/compat' | paste -sd: -)"
export LD_LIBRARY_PATH

require_cuda
report_env

MODE="${MODE:-both}"
UPSTREAM_DIR="${UPSTREAM_DIR:-/data/user/jzou521/codes/cuda/flash-vla/artifacts/upstreams/lingbot-vla}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/user/jzou521/models/lingbot-vla-4b-posttrain-robotwin-fb71a2c}"
QWEN_DIR="${QWEN_DIR:-/data/user/jzou521/codes/cuda/flash-vla/artifacts/upstreams/qwen2.5-vl-3b-instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/user/jzou521/codes/cuda/flash-vla/artifacts/onboarding/lingbot-vla-4b-h100-bf16/official}"

MODES=("${MODE}")
if [[ "${MODE}" == "both" ]]; then
  MODES=(eager compile)
fi

for mode in "${MODES[@]}"; do
  "${PYTHON}" -u -m eval.lingbot.reference \
    --upstream "${UPSTREAM_DIR}" \
    --checkpoint "${CHECKPOINT_DIR}" \
    --qwen "${QWEN_DIR}" \
    --output "${OUTPUT_DIR}" \
    --mode "${mode}" \
    --warmup "${WARMUP:-1}" \
    --reps "${REPS:-1}"
done
