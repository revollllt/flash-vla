#!/bin/bash
# Latency, per-call-site profile and floor of one Target on one plan, reports
# persisted as JSON under profiles/<target>/.
#
#   sbatch sbatch/profile.sh
#   TARGET=h100/pi0 sbatch sbatch/profile.sh
#   PLAN=attn-ffn-cuda-fused-producer-pdl REPS=100 CAPTURE_TRACES=1 sbatch sbatch/profile.sh
#
# Latency runs first: the profiler perturbs scheduling, so a latency claim must
# not come from a process that has already profiled. Each command builds its
# own engine; later builds are cheap because TileLang's cache is warm.
#SBATCH --job-name=profile
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/third_party/cutlass}"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
TARGET="${TARGET:-h100/pi05}"
PLAN="${PLAN:-attn-ffn-cuda-fused-producer-pdl}"
REPS="${REPS:-100}"
TAG="${SLURM_JOB_ID:-local}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/profiles/$(echo "${TARGET}" | tr '/' '_')}"
mkdir -p "${OUT_DIR}"
require_cuda
report_env
plan_args=()
if [[ -n "${PLAN}" ]]; then plan_args=(--plan "${PLAN}"); fi
trace_args=()
if [[ -n "${CAPTURE_TRACES:-}" ]]; then trace_args=(--trace-dir "${OUT_DIR}/traces_${TAG}"); fi
echo "[job] target=${TARGET} plan=${PLAN} out=${OUT_DIR} started $(date)"
echo "== latency (calibration: three legs of the same plan)"
"${PYTHON}" -u -m benchmarks latency --target "${TARGET}" "${plan_args[@]}" --calibrate --reps "${REPS}" --out "${OUT_DIR}/latency_${TAG}.json" > /dev/null
echo "== profile"
"${PYTHON}" -u -m benchmarks profile --target "${TARGET}" "${plan_args[@]}" "${trace_args[@]}" --out "${OUT_DIR}/profile_${TAG}.json"
echo "== floor"
"${PYTHON}" -u -m benchmarks floor --target "${TARGET}" "${plan_args[@]}" --out "${OUT_DIR}/floor_${TAG}.json" > /dev/null
echo "[job] finished $(date)"
