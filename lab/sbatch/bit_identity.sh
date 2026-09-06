#!/bin/bash
# Cross-revision bit identity on one node: dump every declared stage output of
# each Target on each plan from BASE_TREE and from HEAD_TREE, compare the
# pairs with `lab.stage_dump compare`, then run the shallow in-engine
# correctness gate from HEAD_TREE.
#
#   BASE_TREE=artifacts/worktrees/pr0-base sbatch lab/sbatch/bit_identity.sh
#   BASE_TREE=... TARGETS="h100/pi0" PLANS="shipped" sbatch lab/sbatch/bit_identity.sh
#
# HEAD_TREE defaults to the submitting tree. Dumps land under
# artifacts/bit_identity/<job id>/ of the submitting tree.
#SBATCH --job-name=bit-identity
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/third_party/cutlass}"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
HEAD_TREE="$(readlink -f "${HEAD_TREE:-${REPO_DIR}}")"
BASE_TREE="$(readlink -f "${BASE_TREE:?set BASE_TREE to the tree before the change}")"
TARGETS="${TARGETS:-h100/pi05 h100/pi0}"
PLANS="${PLANS:-shipped reference}"
SEED="${SEED:-0}"
OUT="${REPO_DIR}/artifacts/bit_identity/${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}"
require_cuda
report_env
echo "[job] base=${BASE_TREE} ($(git -C "${BASE_TREE}" rev-parse --short HEAD))"
echo "[job] head=${HEAD_TREE} ($(git -C "${HEAD_TREE}" rev-parse --short HEAD))"
echo "[job] started $(date)"
status=0
for target in ${TARGETS}; do
    tag="$(echo "${target}" | tr '/' '_')"
    for plan in ${PLANS}; do
        for side in base head; do
            tree="${BASE_TREE}"; [[ "${side}" == head ]] && tree="${HEAD_TREE}"
            echo "== dump ${side} ${target} ${plan}"
            (cd "${tree}" && PYTHONPATH="${tree}/src:${tree}" "${PYTHON}" -u \
                "${HEAD_TREE}/lab/stage_dump.py" dump --target "${target}" --plan "${plan}" \
                --seed "${SEED}" --out "${OUT}/${tag}_${plan}_${side}.pt")
        done
        echo "== compare ${target} ${plan}"
        (cd "${HEAD_TREE}" && "${PYTHON}" -u -m lab.stage_dump compare \
            "${OUT}/${tag}_${plan}_base.pt" "${OUT}/${tag}_${plan}_head.pt") || status=1
    done
    echo "== correctness gate (head) ${target}: 1 step, 1 layer"
    (cd "${HEAD_TREE}" && "${PYTHON}" -u -m eval.correctness --target "${target}" \
        --steps 1 --layers 1 > "${OUT}/correctness_${tag}_1x1.json") \
        && echo "   passed" || { echo "   FAILED"; status=1; }
done
echo "== smoke (head)"
(cd "${HEAD_TREE}" && "${PYTHON}" -u -m eval.smoke) || status=1
echo "[job] finished $(date) status=${status}"
exit "${status}"
