#!/bin/bash
#SBATCH --job-name=plan-e2e
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
#
# One candidate plan, end to end: the generic in-engine correctness gate
# against the Target's reference plan, then the latency A/B/A in one process
# on one node.
#   sbatch sbatch/plan_e2e.sh
#   PLAN=lab/plans/pi05-attn-cuda.json E2E_REPS=100 sbatch sbatch/plan_e2e.sh
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/third_party/cutlass}"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
export FLASH_VLA_BUILD_VERBOSE=1
TARGET="${TARGET:-h100/pi05}"
PLAN="${PLAN:-shipped}"
E2E_REPS="${E2E_REPS:-100}"
# Space-separated run order for the latency comparison; default keeps the A/B/A.
E2E_PLANS="${E2E_PLANS:-reference ${PLAN} reference}"

require_cuda
report_env
pin_gpu_clocks
echo "[job] started $(date) plan=${PLAN}"
echo "== in-engine correctness: 1 step, 1 layer (gate)"
"${PYTHON}" -u -m eval.correctness --target "${TARGET}" --plan "${PLAN}" --steps 1 --layers 1
echo "== in-engine correctness: 1 step, full depth (report)"
"${PYTHON}" -u -m eval.correctness --target "${TARGET}" --plan "${PLAN}" --steps 1 --layers 0 || true
echo "== in-engine correctness: full depth, isolated (report)"
"${PYTHON}" -u -m eval.correctness --target "${TARGET}" --plan "${PLAN}" --steps 0 --layers 0 --isolate || true
echo "== latency A/B/A"
plan_args=()
for p in ${E2E_PLANS}; do plan_args+=(--plan "${p}"); done
"${PYTHON}" -u -m benchmarks latency --target "${TARGET}" --reps "${E2E_REPS}" "${plan_args[@]}"
echo "[job] finished $(date)"
