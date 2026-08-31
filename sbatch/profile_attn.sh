#!/bin/bash
# Nsight Compute capture of the attention task-loop kernel on one H100.
#   sbatch -w ACD1-21 sbatch/profile_attn.sh            # ncu-capable nodes only:
#   ACD1-10 / 20 / 21 / 31 / 40 / 62 (see server-usage)
# Captures one fused launch per mode in ATTN_MODES (default qkv,attn) with
# --set full, then exports the details page as text next to the report.
# Diagnostic only (rules/performance-and-validation.md): never a latency claim.
#SBATCH --job-name=attn-ncu
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-/data/user/jzou521/codes/cuda/cutlass}"
require_cuda
report_env
mkdir -p "${REPO_DIR}/profiles/attn"
# ATTN_IMPL=fused (default): one task-loop launch per mode in ATTN_MODES.
# ATTN_IMPL=standalone: the standalone kernels of one block invocation,
# filtered by ATTN_KERNEL_REGEX (default: every attn:: kernel, 5 launches).
IMPL="${ATTN_IMPL:-fused}"
MODES="${ATTN_MODES:-qkv,attn}"
if [[ "${IMPL}" == "fused" ]]; then
  REGEX="attn_taskloop_kernel"; COUNT=$(( $(echo "${MODES}" | tr ',' '\n' | wc -l) ))
else
  REGEX="${ATTN_KERNEL_REGEX:-attn::}"; COUNT="${ATTN_COUNT:-5}"
fi
OUT="${REPO_DIR}/profiles/attn/ncu_${IMPL}_${SLURM_JOB_ID}"
ncu --set=full --clock-control=none --target-processes=all --kernel-name-base=function \
    --kernel-name="regex:${REGEX}" --launch-skip=0 --launch-count="${COUNT}" \
    --force-overwrite --export="${OUT}" \
    "${PYTHON}" -u eval/correctness/pi05/attention_block_parity.py \
      --impl "${IMPL}" --modes "${MODES}" --alias alias --replay-check 0 || echo "[ncu] non-zero exit; report may still exist"
ncu -i "${OUT}.ncu-rep" --page details > "${OUT}.details.txt" 2>&1 || true
ncu -i "${OUT}.ncu-rep" --page raw --csv > "${OUT}.raw.csv" 2>&1 || true
echo "[ncu] wrote ${OUT}.ncu-rep / .details.txt / .raw.csv"
