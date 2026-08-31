#!/bin/bash
# Attention-mainloop ablations: the same harness, four builds, timeline only.
#   sbatch sbatch/attn_ablation.sh
#SBATCH --job-name=attn-abl
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-/data/user/jzou521/codes/cuda/cutlass}"
require_cuda
report_env
for d in "" "-DATTN_ABL_NO_SOFTMAX" "-DATTN_ABL_NO_PV" "-DATTN_ABL_NO_S"; do
  echo "[ablation] defines='${d}'"
  ATTN_NVCC_DEFINES="${d}" "${PYTHON}" -u eval/correctness/pi05/attention_block_parity.py \
    --impl fused --modes attn --alias alias --replay-check 0 --timeline --force-timeline --stage-bench --reps 30 \
    2>&1 | grep -E "\[timeline\] attn |\[stage\] attn|\[gate\] worst|Traceback|Error" || true
done
echo "[job] finished $(date)"
