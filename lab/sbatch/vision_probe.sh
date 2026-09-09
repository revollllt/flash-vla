#!/bin/bash
# Vision-lane probe runner: like run.sbatch but exports the tokenizer the
# engine build needs (run.sbatch does not).
#   sbatch lab/sbatch/vision_probe.sh artifacts/ktasks/vision-ln-attn/fc2_cfg.py [args...]
#SBATCH --job-name=vision-probe
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
export CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/third_party/cutlass}"
require_cuda
report_env
echo "[job] started $(date): $*"
"${PYTHON}" -u "$@"
echo "[job] finished $(date)"
