#!/bin/bash
# Per-stage latency and per-kernel breakdown for the whole Pi0.5 pipeline, with
# both reports persisted as JSON under profiles/pi05/.
#
# Usage:
#   sbatch sbatch/profile_pi05.sh
#   CAPTURE_ONLY=profile sbatch sbatch/profile_pi05.sh      # skip the e2e pass
#   CAPTURE_TRACES=1     sbatch sbatch/profile_pi05.sh      # + Chrome traces
#
# This exists because every recorded pi0.5 number lives only in PLAN.md prose:
# the benchmarks print their report and nothing has ever kept it, so a citation
# cannot be checked and a regression cannot be seen. The reports land beside a
# manifest carrying node, driver, torch and git revision, per the repository's
# performance-and-validation rule.
#
# The prefix stage is the reason to run this: PLAN 4.8 tables the decoder kernel
# by kernel and records exactly one prefix kernel, leaving ~45% of that stage
# with no per-kernel attribution at all.
#SBATCH --job-name=pi05-profile
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"

export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"

require_cuda
report_env

# Clocks are not lockable for this user on this partition, so the constants file
# records a 6% noise floor. Try anyway -- pin_gpu_clocks warns and continues --
# and treat any delta smaller than that as noise.
pin_gpu_clocks

OUT_DIR="${OUT_DIR:-${REPO_DIR}/profiles/pi05}"
# `cond && append` would make the loop's exit status the last test's, which is
# a trap under `set -e` for whoever adds a line after it. Use explicit blocks,
# as profile_ffn.sh does.
ARGS=(--out-dir "${OUT_DIR}" --tag "${SLURM_JOB_ID:-local}")
if [[ -n "${CAPTURE_ONLY:-}" ]]; then
    ARGS+=(--only "${CAPTURE_ONLY}")
fi
if [[ -n "${CAPTURE_TRACES:-}" ]]; then
    ARGS+=(--trace-dir "${OUT_DIR}/traces_${SLURM_JOB_ID:-local}")
fi
for opt in num-views chunk-size steps layers reps top seed plan; do
    var="CAPTURE_$(echo "${opt}" | tr 'a-z-' 'A-Z_')"
    if [[ -n "${!var:-}" ]]; then
        ARGS+=("--${opt}" "${!var}")
    fi
done

echo "[job] capturing to ${OUT_DIR}"
echo "[job] started $(date)"
"${PYTHON}" -u "${REPO_DIR}/sbatch/capture_pi05_reports.py" "${ARGS[@]}"
echo "[job] finished $(date)"
