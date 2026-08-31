#!/bin/bash
# qkv traffic ablations on the standalone qkv kernel: baseline / no x TMA / no W TMA,
# with the standalone per-task timeline.  Full log (no filtering).
#SBATCH --job-name=attn-qkvabl
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
for d in "" "-DATTN_ABL_QKV_NO_X -DATTN_ABL_QKV_NO_W" "-DATTN_ABL_QKV_NO_X -DATTN_ABL_QKV_NO_W -DATTN_ABL_QKV_NO_SCALE" "-DATTN_ABL_QKV_NO_X -DATTN_ABL_QKV_NO_W -DATTN_ABL_QKV_NO_MMA" "-DATTN_ABL_QKV_NO_X -DATTN_ABL_QKV_NO_W -DATTN_ABL_QKV_NO_SCALE -DATTN_ABL_QKV_NO_MMA"; do
  echo "[ablation] defines='${d}'"
  ATTN_NVCC_DEFINES="${d}" "${PYTHON}" -u eval/correctness/pi05/attention_block_parity.py \
    --impl standalone --alias alias --replay-check 0 --timeline --force-timeline --op-bench --force-bench --reps 30 --rounds 2 \
    2>&1 | grep -E "\[gate\]|\[sa-timeline\]|\[op\] qkv|\[phase\] built|Traceback|Error" || true
done
echo "[job] finished $(date)"
