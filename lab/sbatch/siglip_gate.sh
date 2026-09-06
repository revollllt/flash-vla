#!/bin/bash
# Lane B promotion evidence: the acceptance gate on the deployed configuration.
#
#   cd <worktree> && sbatch -J laneB-gate lab/sbatch/siglip_gate.sh
#
# The candidate plans are the Target's shipped route plus the vision change, and
# the reference leg is `shipped`, so the A/B/A measures exactly what deployment
# would see rather than the change against the all-TileLang reference route.
#SBATCH --job-name=laneB-gate
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/third_party/cutlass}"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
REPS="${REPS:-100}"
WS="${WS:-${REPO_DIR}/artifacts/ktasks/siglip}"
TAG="${SLURM_JOB_ID:-local}"
mkdir -p "${WS}/runs/gate_${TAG}"
require_cuda
report_env
echo "[job] node=$(hostname -s) revision=$(git -C "${REPO_DIR}" rev-parse --short HEAD) started $(date)"

# Pi0 carries the promotable candidate; Pi0.5's best is recorded for its
# verdict even though the A/B/A already puts it under the bar.
for spec in "h100/pi0 lab/plans/pi0-siglip-shipped-cuda.json" \
            "h100/pi05 lab/plans/pi05-siglip-shipped-attn.json"; do
  set -- ${spec}
  target="$1"; plan="$2"
  echo "== eval.gate ${target} ${plan} against shipped"
  "${PYTHON}" -u -m eval.gate --target "${target}" --candidate "${plan}" \
    --reference shipped --baseline --reps "${REPS}" \
    --out-dir "${WS}/runs/gate_${TAG}" || echo "[job] gate exit $? for ${target} (verdict is in the record)"
done
echo "[job] finished $(date)"
