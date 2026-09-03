#!/bin/bash
# Same-node A/B/A of the encoder attention route: torch chain vs the fused
# CUDA kernel, three e2e passes in one job so node and clock drift cancel.
#   sbatch -w ACD1-33 sbatch/mqa_ab.sh
#SBATCH --job-name=mqa-ab
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/third_party/cutlass}"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
require_cuda
report_env
REPS="${E2E_REPS:-30}"
PLAN="${AB_PLAN:-attn-ffn-cuda-fused-producer-pdl}"
# The route is read at import, so each leg is its own process.
for route in torch cuda torch; do
  echo "== leg route=${route}"
  ENC_ATTN_ROUTE="${route}" "${PYTHON}" -u -m benchmarks.e2e_pi05 --reps "${REPS}" --plan "${PLAN}" \
    | grep -E "stage|prefix|vision|decoder|forward|min_ms|median_ms" || true
done
echo "[job] finished $(date)"
