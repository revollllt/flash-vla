#!/bin/bash
# Lane B (SigLIP vision) candidate bundle: baselines, T2 parity, the in-engine
# correctness gate and the same-node A/B/A, for both Targets in one job.
#
#   cd <worktree> && BASELINES=1 sbatch -J laneB-j0 lab/sbatch/siglip.sh
#   cd <worktree> && CAND=siglip-cuda sbatch -J laneB-j1 lab/sbatch/siglip.sh
#
# CAND names the candidate plans as lab/plans/{pi05,pi0}-${CAND}.json. Baselines
# run first and only when asked: a candidate's number is never read before the
# baseline it is compared against exists (kernel-design, step 2).
#SBATCH --job-name=laneB-siglip
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/third_party/cutlass}"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
export FLASH_VLA_BUILD_VERBOSE="${FLASH_VLA_BUILD_VERBOSE:-1}"

CAND="${CAND:-siglip-cublas}"
BACKEND="${BACKEND:-${CAND}}"
REPS="${REPS:-100}"
WS="${WS:-${REPO_DIR}/artifacts/ktasks/siglip}"
TAG="${SLURM_JOB_ID:-local}"
mkdir -p "${WS}/runs"

require_cuda
report_env
echo "[job] node=$(hostname -s) cand=${CAND} reps=${REPS} started $(date)"
echo "[job] revision=$(git -C "${REPO_DIR}" rev-parse --short HEAD)"

# Clocks stay unlocked: this partition grants no permission to pin them, and the
# registry's candidate rule reads `min` for exactly that reason.

if [[ -n "${BASELINES:-}" ]]; then
  for target in h100/pi05 h100/pi0; do
    for plan in shipped reference; do
      echo "== baseline kernels ${target} ${plan}"
      "${PYTHON}" -u -m benchmarks kernels --target "${target}" --plan "${plan}" \
        --segment vision_encoder --timer cupti --csv "${WS}/benchmark.csv"
    done
    echo "== floor ${target} (denominator on the current runtime)"
    "${PYTHON}" -u -m benchmarks floor --target "${target}" \
      --out "${WS}/runs/floor_$(echo "${target}" | tr '/' '_')_${TAG}.json" > /dev/null
  done
fi

echo "== T2 parity: ${BACKEND} wrappers against their ABI mirrors"
"${PYTHON}" -u -m lab.siglip.parity --backend "${BACKEND}" \
  | tee "${WS}/runs/parity_${BACKEND}_${TAG}.json"

for target in h100/pi05 h100/pi0; do
  prefix="${target#h100/}"
  plan="lab/plans/${prefix}-${CAND}.json"
  [[ -f "${plan}" ]] || { echo "[job] no plan ${plan}; skipping ${target}"; continue; }

  echo "== in-engine correctness ${target} ${plan}: 1 step, 1 layer (GATE)"
  "${PYTHON}" -u -m eval.correctness --target "${target}" --plan "${plan}" --steps 1 --layers 1
  echo "== in-engine correctness ${target}: 1 step, full depth (report)"
  "${PYTHON}" -u -m eval.correctness --target "${target}" --plan "${plan}" --steps 1 --layers 0 || true

  echo "== isolated per-call-site ${target} ${plan}"
  "${PYTHON}" -u -m benchmarks kernels --target "${target}" --plan "${plan}" \
    --segment vision_encoder --timer cupti --csv "${WS}/benchmark.csv"

  echo "== A/B/A ${target}: reference / ${CAND} / reference, ${REPS} reps, same process"
  "${PYTHON}" -u -m benchmarks latency --target "${target}" --reps "${REPS}" \
    --plan reference --plan "${plan}" --plan reference \
    --out "${WS}/runs/aba_${prefix}_${CAND}_${TAG}.json"
done

echo "[job] finished $(date)"
