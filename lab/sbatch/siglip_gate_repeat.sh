#!/bin/bash
#SBATCH --job-name=laneB-gate2
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
WS="${REPO_DIR}/artifacts/ktasks/siglip"; TAG="${SLURM_JOB_ID:-local}"
require_cuda; report_env
echo "[job] node=$(hostname -s) revision=$(git -C "${REPO_DIR}" rev-parse --short HEAD)"
# The Pi0.5 verdict in job 600039 passed by 0.034 ms with a 0.082 ms control
# spread, so the same run is repeated twice to see whether the pass reproduces
# or was the first control leg.
for i in 1 2; do
  mkdir -p "${WS}/runs/gate_${TAG}_r${i}"
  echo "== repeat ${i}: eval.gate h100/pi05 pi05-siglip-shipped-attn against shipped"
  "${PYTHON}" -u -m eval.gate --target h100/pi05 \
    --candidate lab/plans/pi05-siglip-shipped-attn.json --reference shipped \
    --baseline --reps 100 --seed "${i}" --out-dir "${WS}/runs/gate_${TAG}_r${i}" \
    || echo "[job] gate exit $? (verdict is in the record)"
done
echo "[job] finished $(date)"
