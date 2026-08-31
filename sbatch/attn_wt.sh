#!/bin/bash
# W_qkv major experiment: as-stored (MN-major B) vs pre-transposed (K-major B).
#SBATCH --job-name=attn-wt
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
for d in "" "-DATTN_QKV_WT"; do
  echo "[variant] defines='${d}'"
  ATTN_NVCC_DEFINES="${d}" "${PYTHON}" -u eval/correctness/pi05/attention_block_parity.py \
    --impl standalone --alias alias --replay-check 1 --timeline --op-bench --reps 30 --rounds 2 \
    2>&1 | grep -E "\[gate\] (worst|replay|standalone.*q_buf)|\[sa-timeline\] qkv|\[op\] qkv|\[phase\] built|Traceback|Error" || true
done
echo "[job] finished $(date)"
