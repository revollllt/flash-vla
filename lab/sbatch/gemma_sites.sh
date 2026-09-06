#!/bin/bash
# Lane C: per-call-site timing of the backbone on both plans, both Targets.
#
#   sbatch lab/sbatch/gemma_sites.sh
#   PLANS="shipped lab/plans/pi0-gemma-cuda.json" TARGETS=pi0 sbatch lab/sbatch/gemma_sites.sh
#
# Why this exists as its own job. `benchmarks kernels --segment llm_backbone`
# cannot run on Pi0: the segment's first call site is `llm_backbone_embed_prompt`,
# which at Pi0's `prompt_len = 0` copies a (0, 2048) tensor, issues no CUDA
# kernel at all, and the CUPTI timer raises "No kernel activities recorded for
# an iteration". The floor model already tolerates the degenerate site (it
# records `measured_us: null`); the kernel timer does not. Naming the five real
# call sites explicitly is the narrowest way round it and needs no harness
# change.
#
# What it answers: candidate C0 moved Pi0's backbone segment by 0.23 ms and its
# chunk `min` by 0.096 ms, against 0.66 ms predicted from the floor report's
# per-site attributed times. Either those three sites did not get as much
# faster as predicted, or they did and the segment did not follow. Only a
# same-job per-site measurement on both plans separates the two, and the answer
# recalibrates every remaining candidate in this lane.
#SBATCH --job-name=laneC-sites
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -uo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/third_party/cutlass}"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
export FLASH_VLA_BUILD_VERBOSE="${FLASH_VLA_BUILD_VERBOSE:-1}"

TARGETS="${TARGETS:-pi0 pi05}"
TIMERS="${TIMERS:-cupti cudagraph}"

WS="${REPO_DIR}/artifacts/ktasks/gemma_backbone"
OUT="${WS}/runs/sites_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}"

# The five real backbone call sites. `llm_backbone_embed_prompt` and
# `llm_backbone_projector` are excluded: the first is degenerate on Pi0 and
# neither is in this lane's scope.
SITE_ARGS=(--site llm_backbone_norm_qkv_rope
           --site llm_backbone_attention
           --site llm_backbone_out_proj_residual
           --site llm_backbone_norm_gated_ffn
           --site llm_backbone_ffn_down_residual)

require_cuda
report_env
echo "[job] rev=$(git -C "${REPO_DIR}" rev-parse --short HEAD)"
echo "[job] started $(date)"

for t in ${TARGETS}; do
    for plan in shipped "lab/plans/${t}-gemma-cuda.json"; do
        slug=$(basename "${plan}" .json)
        for timer in ${TIMERS}; do
            echo
            echo "=================== ${t} ${slug} ${timer} ==================="
            "${PYTHON}" -u -m benchmarks kernels --target "h100/${t}" --plan "${plan}" \
                "${SITE_ARGS[@]}" --timer "${timer}" \
                --csv "${OUT}/sites_${t}_${slug}_${timer}.csv" \
                2>&1 | tee "${OUT}/sites_${t}_${slug}_${timer}.log"
            echo "!! ${t} ${slug} ${timer} exit ${PIPESTATUS[0]}"
        done
    done
done

echo
echo "[job] artifacts in ${OUT}"
echo "[job] finished $(date)"
