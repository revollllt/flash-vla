#!/bin/bash
# Profile the minimal 132-CTA FFN task-loop on one H100.
# Usage:
#   PROFILE_KIND=nsys PROFILE_OUT=profiles/ffn/nsys_%j sbatch sbatch/profile_ffn.sh
#   PROFILE_KIND=ncu  PROFILE_OUT=profiles/ffn/ncu_%j  sbatch sbatch/profile_ffn.sh
#
# The workload runs GU, DR and full once.  The profiler filters/captures only
# ffn_taskloop_kernel; parity remains the correctness gate before profiling.
# The output directory is intentionally outside .cache so reports survive
# compiler-cache cleanup.
#SBATCH --job-name=ffn-profile
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"

export CUTLASS_DIR="${CUTLASS_DIR:-/data/user/jzou521/codes/cuda/cutlass}"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
export FLASH_VLA_BUILD_VERBOSE="${FLASH_VLA_BUILD_VERBOSE:-1}"

require_cuda
report_env
mkdir -p "${REPO_DIR}/profiles/ffn"

PROFILE_KIND="${PROFILE_KIND:-nsys}"
PROFILE_OUT="${PROFILE_OUT:-${REPO_DIR}/profiles/ffn/${PROFILE_KIND}_${SLURM_JOB_ID}}"
PROFILE_OUT="${PROFILE_OUT//%j/${SLURM_JOB_ID:-local}}"
PY_ARGS=(eval/correctness/pi05/ffn_taskloop_parity.py
         --modes gu,dr,full --replay-check 0 --seed "${PROFILE_SEED:-7}")

echo "[profile] kind=${PROFILE_KIND} output=${PROFILE_OUT}"
echo "[profile] started $(date)"

case "${PROFILE_KIND}" in
  nsys)
    # CUDA timeline and kernel launch/stream visibility.  Disable CPU sampling
    # because this workload is GPU-bound and sampling adds noise to the short
    # FFN launch.
    nsys profile \
      --trace=cuda,nvtx,osrt \
      --sample=none \
      --cpuctxsw=none \
      --cuda-memory-usage=false \
      --force-overwrite=true \
      --output="${PROFILE_OUT}" \
      "${PYTHON}" -u "${PY_ARGS[@]}"
    ;;
  ncu)
    # Capture the three FFN launches (GU, DR, full) and ignore torch reference
    # kernels. `basic` contains launch stats, occupancy, speed-of-light and
    # workload distribution; a later `--set full` can
    # be run on a selected launch if a stall category needs expansion.
    NCU_ARGS=(
      "--set=${NCU_SET:-basic}"
      --clock-control=none
      --target-processes=all
      --kernel-name-base=function
      --kernel-name=regex:ffn_taskloop_kernel
      "--launch-skip=${NCU_LAUNCH_SKIP:-0}"
      "--launch-count=${NCU_LAUNCH_COUNT:-3}"
      --force-overwrite
      "--export=${PROFILE_OUT}"
    )
    if [[ -n "${NCU_SECTIONS:-}" ]]; then
      IFS=',' read -r -a _sections <<< "${NCU_SECTIONS}"
      for _section in "${_sections[@]}"; do
        NCU_ARGS+=("--section=${_section}")
      done
    fi
    ncu "${NCU_ARGS[@]}" "${PYTHON}" -u "${PY_ARGS[@]}"
    ;;
  *)
    echo "unknown PROFILE_KIND=${PROFILE_KIND}; expected nsys or ncu" >&2
    exit 2
    ;;
esac

echo "[profile] finished $(date)"
