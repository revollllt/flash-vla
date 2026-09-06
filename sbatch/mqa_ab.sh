#!/bin/bash
# Same-node A/B/A of the encoder attention route: the fused CUDA kernel against
# the reference torch chain, three e2e passes in ONE process so node and clock
# drift cancel and the control leg has to reproduce.
#   sbatch -w ACD1-33 sbatch/mqa_ab.sh
# The two legs are call-site plans that differ in exactly one entry, so each
# report records which route produced it (the plan in its identity).
#
# Read this design for provenance, not for the magnitude of a claim. One job,
# one process, and a recorded plan per leg naming the route -- which the
# import-time switch it replaced could not do. But its control legs drifted
# 0.214 ms in job 591067 while the plan-invariant vision stage moved only
# 0.015 ms, so the spread is prefix-specific and wider than the effect it would
# measure; the promoted -0.156 ms prefix number comes from a THREE-PROCESS
# A/B/A (job 589207) whose control legs reproduced exactly. Untested
# hypothesis for the difference: the prefix is the only stage that allocates
# inside graph capture on the torch leg, and three engines built back to back
# leave the caching allocator in a different state for the third. Until that is
# measured, take a magnitude claim for this call site from separate processes.
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
REPS="${E2E_REPS:-100}"
CUDA_PLAN="${AB_PLAN:-lab/plans/pi05-attn-ffn-cuda-fused-producer.json}"
CHAIN_PLAN="${AB_PLAN_REFERENCE:-lab/plans/pi05-attn-ffn-cuda-fused-producer-enc-tilelang.json}"
"${PYTHON}" -u -m benchmarks latency --target h100/pi05 --reps "${REPS}" \
    --plan "${CUDA_PLAN}" --plan "${CHAIN_PLAN}" --plan "${CUDA_PLAN}"
echo "[job] finished $(date)"
