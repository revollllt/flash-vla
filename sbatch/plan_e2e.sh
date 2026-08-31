#!/bin/bash
#SBATCH --job-name=plan-e2e
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
#
# Mixed call-site plan, end to end: parity gate of the `cuda` attention half
# against the all-TileLang decoder, then the e2e benchmark for both plans in
# one process (A/B/A) on one node.
#   sbatch sbatch/plan_e2e.sh
#   PLAN=attn-cuda E2E_REPS=30 sbatch sbatch/plan_e2e.sh
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-/data/user/jzou521/codes/cuda/cutlass}"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
export FLASH_VLA_BUILD_VERBOSE=1
PLAN="${PLAN:-attn-cuda}"
E2E_REPS="${E2E_REPS:-30}"
# Space-separated run order for the e2e comparison; default keeps the A/B/A.
E2E_PLANS="${E2E_PLANS:-tilelang ${PLAN} tilelang}"

require_cuda
report_env
pin_gpu_clocks
echo "[job] started $(date) plan=${PLAN}"
echo "== parity: 1 step, 1 layer"
"${PYTHON}" -u -m eval.correctness.pi05.plan_parity --plan "${PLAN}" --steps 1 --layers 1
echo "== parity: 1 step, 18 layers"
"${PYTHON}" -u -m eval.correctness.pi05.plan_parity --plan "${PLAN}" --steps 1 --layers 18
echo "== parity (reported, not gated): 10 steps, 18 layers"
"${PYTHON}" -u -m eval.correctness.pi05.plan_parity --plan "${PLAN}" --steps 10 --layers 18 || true
echo "== e2e A/B/A"
plan_args=()
for p in ${E2E_PLANS}; do plan_args+=(--plan "${p}"); done
"${PYTHON}" -u -m benchmarks.e2e_pi05 --reps "${E2E_REPS}" "${plan_args[@]}"
echo "[job] finished $(date)"
