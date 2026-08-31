#!/bin/bash
#SBATCH --job-name=pi05-cuda
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
#
# Run one command against the Pi0.5 target on a GPU node.
#   sbatch sbatch/pi05_cuda.sh -m eval.correctness.pi05.kernel_parity --only qkv
# Everything after the script name is passed to the interpreter verbatim.
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"

# One interpreter for this target: the cuteDSL venv, torch 2.11.0+cu130 with
# tilelang 0.1.11. `_common.sh` already defaults to it and honours a caller
# override, so nothing is reassigned here -- an assignment after the source
# would be a no-op anyway, which is exactly the bug this comment replaces.
# `sentencepiece` was added to that venv on 2026-08-20 for Pi0.5 tokenization.
#
# Every number in specs/tile/pi05-decoder-fused-cuda.md must come from this
# build. Mixing builds makes a kernel look faster or slower than its baseline
# for reasons that have nothing to do with the kernel.
export CUTLASS_DIR="${CUTLASS_DIR:-/data/user/jzou521/codes/cuda/cutlass}"
# Pi0.5 tokenizes on the host, so anything touching the full pass needs the real
# PaliGemma tokenizer. Kernel-level parity runs do not, which is why a missing
# one shows up only at e2e -- set it here so both paths behave the same.
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
export FLASH_VLA_BUILD_VERBOSE="${FLASH_VLA_BUILD_VERBOSE:-1}"

require_cuda
report_env
pin_gpu_clocks
echo "[job] running: ${PYTHON} -u $*"
echo "[job] started $(date)"
"${PYTHON}" -u "$@"
echo "[job] finished $(date)"
