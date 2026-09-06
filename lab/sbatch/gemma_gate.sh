#!/bin/bash
# Lane C: the promotion gate for the shared backbone package, both Targets.
#
#   sbatch lab/sbatch/gemma_gate.sh
#   PLAN_SUFFIX=gemma-cuda MODE_PI0=improve sbatch lab/sbatch/gemma_gate.sh
#
# Two Targets, two modes, and the modes differ for a reason. On Pi0 the
# candidate really changes the program -- the backbone attention and the output
# projection move to the component package -- so it is judged under `improve`,
# the registry's rule for a performance candidate. On Pi0.5 the same plan is
# proven bit-identical to shipped (`lab.stage_dump compare`, job 599788): the
# kernel moved into the package and nothing else, which is a refactor, so it is
# judged under `no-regression`. Asking `improve` of a bit-identical change
# would be asking it to be faster than itself.
#
# `--baseline` runs the official-baseline tier under the OpenPI interpreter the
# registry names. Both Targets can now reach it: Pi0's checkpoint fix is on
# this branch (82d2f33) and Pi0.5's chunk tail no longer violates the
# deployment bound (277a58d, lane A).
#SBATCH --job-name=laneC-gate
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -uo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/third_party/cutlass}"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
export FLASH_VLA_BUILD_VERBOSE=1

PLAN_SUFFIX="${PLAN_SUFFIX:-gemma-cuda}"
MODE_PI0="${MODE_PI0:-improve}"
MODE_PI05="${MODE_PI05:-no-regression}"
REPS="${REPS:-100}"
WS="${REPO_DIR}/artifacts/ktasks/gemma_backbone"
OUT="${WS}/runs/gate_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}"

require_cuda
report_env
echo "[job] rev=$(git -C "${REPO_DIR}" rev-parse --short HEAD)"
echo "[job] started $(date)"

gate() {
    local t="$1" mode="$2"
    echo; echo "=================== gate ${t} (${mode}) ==================="
    "${PYTHON}" -u -m eval.gate --target "h100/${t}" \
        --candidate "lab/plans/${t}-${PLAN_SUFFIX}.json" \
        --mode "${mode}" --baseline --reps "${REPS}" \
        --out-dir "${OUT}" 2>&1 | tee "${OUT}/gate_${t}.log"
    echo "!! gate ${t} exit ${PIPESTATUS[0]}  (0 pass / 1 fail / 2 blocked)"
}

gate pi0 "${MODE_PI0}"
gate pi05 "${MODE_PI05}"

echo
echo "[job] artifacts in ${OUT}"
echo "[job] finished $(date)"
