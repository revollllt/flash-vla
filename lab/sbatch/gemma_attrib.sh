#!/bin/bash
# Lane C: does a kernel-time saving in the backbone transfer to wall time?
#
#   sbatch lab/sbatch/gemma_attrib.sh
#
# This is the deciding experiment for the rest of the lane, not a measurement
# of C0. Candidate C0 removed 0.337 ms of backbone kernel time on Pi0 (the
# `cudagraph` per-site timer, which reproduces the in-graph attribution to
# ~1 % at every Pi0.5 site) and moved the chunk by 0.10-0.12 ms on `min`,
# `median` and `p99` alike. Meanwhile `benchmarks profile` says the backbone
# segment is 98-99 % kernel time (Pi0.5 wall 6313 us against 6246 attributed;
# Pi0 5728 against 5625 with 178 us of device copies inside the wall), so
# there is no gap for a saving to disappear into.
#
# Both cannot be true, and which one is wrong decides whether C1 is worth
# building: C1's thesis is worth 0.6 ms of gated-FFN KERNEL time, which is
# either 0.6 ms of chunk or 0.2 ms of chunk depending on the answer.
#
# The measurement: per-call-site in-graph attribution on BOTH plans of BOTH
# Targets, in one job. The shipped side reproduces job 599808 on a possibly
# different node; the candidate side is the number that has never been taken.
# Subtracting them gives the in-graph saving directly, with no timer standing
# in for the pipeline.
#SBATCH --job-name=laneC-attrib
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -uo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/third_party/cutlass}"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
export FLASH_VLA_BUILD_VERBOSE="${FLASH_VLA_BUILD_VERBOSE:-1}"

TARGETS="${TARGETS:-pi0 pi05}"
REPS="${REPS:-100}"
WS="${REPO_DIR}/artifacts/ktasks/gemma_backbone"
OUT="${WS}/runs/attrib_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}"

require_cuda
report_env
echo "[job] rev=$(git -C "${REPO_DIR}" rev-parse --short HEAD)"
echo "[job] started $(date)"

for t in ${TARGETS}; do
    for plan in shipped "lab/plans/${t}-gemma-cuda.json"; do
        slug=$(basename "${plan}" .json)
        echo
        echo "=================== profile ${t} ${slug} ==================="
        "${PYTHON}" -u -m benchmarks profile --target "h100/${t}" --plan "${plan}" \
            --out "${OUT}/profile_${t}_${slug}.json" \
            2>&1 | tee "${OUT}/profile_${t}_${slug}.log"
        echo "!! profile ${t} ${slug} exit ${PIPESTATUS[0]}"
    done
    # A second A/B/A on this node, to see whether C0's 0.10 ms reproduces off
    # ACD1-58. A same-node A/B/A is the promotion instrument; a second one on
    # another node is what says the effect is the kernel and not the node.
    echo
    echo "=================== aba ${t} (repeat on this node) ==================="
    "${PYTHON}" -u -m benchmarks latency --target "h100/${t}" --reps "${REPS}" \
        --plan shipped --plan "lab/plans/${t}-gemma-cuda.json" --plan shipped \
        --out "${OUT}/aba_${t}_vs_shipped.json" 2>&1 | tee "${OUT}/aba_${t}_vs_shipped.log"
    echo "!! aba ${t} exit ${PIPESTATUS[0]}"
done

echo
echo "[job] artifacts in ${OUT}"
echo "[job] finished $(date)"
