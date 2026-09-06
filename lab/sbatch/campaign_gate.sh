#!/bin/bash
# The campaign's closing check on one node: each Target's shipped plan gated
# against a named baseline plan (default: the pre-campaign shipped plans
# pinned under lab/plans/<target>-preshipped.json), then a fresh latency,
# profile and floor report of the shipped plan, for the new headroom.
#   sbatch lab/sbatch/campaign_gate.sh
#   TARGETS="h100/pi0" BASELINE="lab/plans/pi0-preshipped.json" sbatch lab/sbatch/campaign_gate.sh
# Reports land under artifacts/campaign/<job id>/ of the submitting tree.
#SBATCH --job-name=campaign-gate
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/third_party/cutlass}"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
TARGETS="${TARGETS:-h100/pi05 h100/pi0}"
REPS="${REPS:-100}"
OUT="${REPO_DIR}/artifacts/campaign/${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}"
require_cuda
report_env
echo "[job] tree $(git rev-parse --short HEAD) started $(date)"
for target in ${TARGETS}; do
    tag="$(echo "${target}" | tr '/' '_')"
    baseline="${BASELINE:-lab/plans/${target#h100/}-preshipped.json}"
    echo "== gate ${target}: shipped against ${baseline}, --baseline, reps ${REPS}"
    "${PYTHON}" -u -m eval.gate --target "${target}" --candidate shipped --reference "${baseline}" \
        --baseline --reps "${REPS}" --out-dir "${OUT}/gate" > "${OUT}/gate_${tag}.log" 2>&1 \
        || echo "   gate exit $? (the verdict is in the record)"
    grep -h "verdict\|\[gate  \]\|\[report\]\|latency:\|deployment:" "${OUT}/gate_${tag}.log" | tail -12 || true
    echo "== latency (calibration), profile, floor: ${target} shipped"
    "${PYTHON}" -u -m benchmarks latency --target "${target}" --plan shipped --calibrate \
        --reps "${REPS}" --out "${OUT}/latency_${tag}.json" > "${OUT}/latency_${tag}.log" 2>&1
    "${PYTHON}" -u -m benchmarks profile --target "${target}" --plan shipped \
        --out "${OUT}/profile_${tag}.json" > "${OUT}/profile_${tag}.log" 2>&1
    "${PYTHON}" -u -m benchmarks floor --target "${target}" --plan shipped \
        --out "${OUT}/floor_${tag}.json" 2>&1 | tail -45
done
echo "[job] finished $(date)"
