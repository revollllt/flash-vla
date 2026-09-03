#!/bin/bash
# Same-node A/B/A of the encoder attention route: the fused CUDA kernel against
# the reference torch chain, three e2e passes in ONE process so node and clock
# drift cancel and the control leg has to reproduce.
#   sbatch -w ACD1-33 sbatch/mqa_ab.sh
# The two legs are call-site plans that differ in exactly one entry, so each
# report records which route produced it (`benchmarks/plans.py`).
#SBATCH --job-name=mqa-ab
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/third_party/cutlass}"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
require_cuda
report_env
REPS="${E2E_REPS:-30}"
CUDA_PLAN="${AB_PLAN:-attn-ffn-cuda-fused-producer}"
CHAIN_PLAN="${AB_PLAN_REFERENCE:-attn-ffn-cuda-fused-producer-enc-tilelang}"
"${PYTHON}" -u -m benchmarks.e2e_pi05 --reps "${REPS}" \
    --plan "${CUDA_PLAN}" --plan "${CHAIN_PLAN}" --plan "${CUDA_PLAN}"
echo "[job] finished $(date)"
