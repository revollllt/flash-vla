#!/bin/bash
# Lane C: one candidate, one job -- parity, correctness, per-site timing, A/B/A.
#
#   TAG=c0 sbatch lab/sbatch/gemma_candidate.sh
#   TAG=c1 TARGETS="pi05 pi0" KERNELS="gated_ffn" sbatch lab/sbatch/gemma_candidate.sh
#   TAG=c0 BIT_IDENTITY=pi05 sbatch lab/sbatch/gemma_candidate.sh
#
# Correctness runs before latency, always: a faster result never overrides a
# numerical mismatch. Everything for one candidate is in one job so the
# per-site timings, the A/B/A legs and the parity numbers share a node -- the
# partition does not pin clocks, so cross-job comparison of these is not
# evidence and same-job comparison is.
#
# Two A/B/A runs per Target, and they answer different questions:
#   shipped / candidate / shipped   what THIS lane changed (the attribution)
#   reference / candidate / reference   what `eval.gate` will apply the
#                                       registry's candidate rule to
# The first is the lane's number; the second is the registry's.
#SBATCH --job-name=laneC-cand
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -uo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/third_party/cutlass}"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
export FLASH_VLA_BUILD_VERBOSE="${FLASH_VLA_BUILD_VERBOSE:-1}"

TAG="${TAG:?set TAG, e.g. TAG=c0}"
TARGETS="${TARGETS:-pi0 pi05}"
#: empty means every kernel the parity harness knows
KERNELS="${KERNELS:-}"
#: a Target named here gets a shipped-vs-candidate bit-identity dump comparison
BIT_IDENTITY="${BIT_IDENTITY:-}"
REPS="${REPS:-100}"

WS="${REPO_DIR}/artifacts/ktasks/gemma_backbone"
OUT="${WS}/runs/${TAG}_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}"

require_cuda
report_env
echo "[job] tag=${TAG} targets='${TARGETS}' rev=$(git -C "${REPO_DIR}" rev-parse --short HEAD)"
echo "[job] started $(date)"

step() { echo; echo "=================== $* ==================="; }
run() {  # run <name> <cmd...>; keeps going so one failure does not hide the rest
    local name="$1"; shift
    step "${name}"
    "$@" 2>&1 | tee "${OUT}/${name}.log"
    local rc=${PIPESTATUS[0]}
    echo "!! ${name} exit ${rc}"
    return 0
}

kernel_args=()
for k in ${KERNELS}; do kernel_args+=(--kernel "${k}"); done

# -- bit identity: a kernel that moved without a numerical change -----------
if [[ -n "${BIT_IDENTITY}" ]]; then
    t="${BIT_IDENTITY}"
    run "bitid_${t}_shipped_dump" "${PYTHON}" -u -m lab.stage_dump dump \
        --target "h100/${t}" --plan shipped --out "${OUT}/bitid_${t}_shipped.pt"
    run "bitid_${t}_candidate_dump" "${PYTHON}" -u -m lab.stage_dump dump \
        --target "h100/${t}" --plan "lab/plans/${t}-gemma-cuda.json" \
        --out "${OUT}/bitid_${t}_candidate.pt"
    run "bitid_${t}_compare" "${PYTHON}" -u -m lab.stage_dump compare \
        "${OUT}/bitid_${t}_shipped.pt" "${OUT}/bitid_${t}_candidate.pt"
fi

for t in ${TARGETS}; do
    plan="lab/plans/${t}-gemma-cuda.json"

    # -- T2 kernel parity: the structural gate ------------------------------
    run "parity_${t}" "${PYTHON}" -u -m lab.gemma_backbone.parity \
        --target "h100/${t}" "${kernel_args[@]}" --json "${OUT}/parity_${t}.json"

    # -- in-engine, against the Target's reference route --------------------
    # layers 1 is the registry's shallow gate; layers 2 is the shallowest
    # depth that runs every backbone call site (the loop breaks after the QKV
    # projection of its last layer); layers 0 is full depth, a report.
    run "correctness_${t}_l1" "${PYTHON}" -u -m eval.correctness \
        --target "h100/${t}" --plan "${plan}" --steps 1 --layers 1
    run "correctness_${t}_l2" "${PYTHON}" -u -m eval.correctness \
        --target "h100/${t}" --plan "${plan}" --steps 1 --layers 2
    run "correctness_${t}_full" "${PYTHON}" -u -m eval.correctness \
        --target "h100/${t}" --plan "${plan}" --steps 1 --layers 0

    # -- per call site, same job, against the incumbent ---------------------
    run "kernels_${t}_shipped" "${PYTHON}" -u -m benchmarks kernels \
        --target "h100/${t}" --plan shipped --segment llm_backbone --timer cupti \
        --csv "${OUT}/kernels_${t}_shipped.csv"
    run "kernels_${t}_candidate" "${PYTHON}" -u -m benchmarks kernels \
        --target "h100/${t}" --plan "${plan}" --segment llm_backbone --timer cupti \
        --csv "${OUT}/kernels_${t}_candidate.csv"

    # -- A/B/A, same process: the lane's number, then the registry's --------
    run "aba_${t}_vs_shipped" "${PYTHON}" -u -m benchmarks latency \
        --target "h100/${t}" --reps "${REPS}" \
        --plan shipped --plan "${plan}" --plan shipped \
        --out "${OUT}/aba_${t}_vs_shipped.json"
    run "aba_${t}_vs_reference" "${PYTHON}" -u -m benchmarks latency \
        --target "h100/${t}" --reps "${REPS}" \
        --plan reference --plan "${plan}" --plan reference \
        --out "${OUT}/aba_${t}_vs_reference.json"
done

echo
echo "[job] artifacts in ${OUT}"
echo "[job] finished $(date)"
