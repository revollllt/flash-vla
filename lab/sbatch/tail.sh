#!/bin/bash
# The Pi0.5 chunk-tail lane in one job: re-baseline, negative control, then the
# four decisive experiments, all on one node so every comparison is same-node.
#
#   sbatch -J laneA-tail lab/sbatch/tail.sh
#   TAIL_STEPS="experiments" TAIL_LEGS=8 sbatch -J laneA-tail2 lab/sbatch/tail.sh
#
# The queue on this partition can take hours, so everything one job can do it
# does: TileLang compiles once into this job's cache and every leg after that
# reuses it. Steps are selectable through TAIL_STEPS so a rerun can skip what
# already has evidence.
#SBATCH --job-name=laneA-tail
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"

# Pi0.5 tokenizes the robot state on the host, so the end-to-end path needs the
# real PaliGemma tokenizer; the CUDA backends need CUTLASS. `sbatch/run.sbatch`
# exports neither.
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
export CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/third_party/cutlass}"
export FLASH_VLA_BUILD_VERBOSE="${FLASH_VLA_BUILD_VERBOSE:-1}"

TAIL_STEPS="${TAIL_STEPS:-rebaseline control experiments}"
TAIL_REPS="${TAIL_REPS:-100}"
TAIL_LEGS="${TAIL_LEGS:-8}"
TAIL_PLANS="${TAIL_PLANS:-shipped reference}"
OUT="${TAIL_OUT:-${REPO_DIR}/artifacts/ktasks/pi05-chunk-tail/runs}/${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}"

require_cuda
report_env
# Deployment does not lock clocks, and `eval/acceptance.py` reads every latency
# number under deployment conditions, so this job does not call pin_gpu_clocks.
echo "[job] node=$(hostname -s) steps='${TAIL_STEPS}' reps=${TAIL_REPS} legs=${TAIL_LEGS}"
echo "[job] out=${OUT}"
echo "[job] started $(date)"

for step in ${TAIL_STEPS}; do
    case "${step}" in
    rebaseline)
        echo "== re-baseline: Pi0.5 A/B/A on the amended runtime, with attribution"
        "${PYTHON}" -u -m benchmarks latency --target h100/pi05 \
            --plan reference --plan shipped --plan reference \
            --reps "${TAIL_REPS}" --out "${OUT}/rebaseline_pi05.json" > "${OUT}/rebaseline_pi05.log"
        # everything before the report itself: the per-leg metrics and the
        # attribution summary. `sed` quits at the report's opening brace.
        sed -n '/^{$/q;p' "${OUT}/rebaseline_pi05.log"
        ;;
    control)
        echo "== negative control: Pi0, the Target with no host slot in its forward"
        "${PYTHON}" -u -m benchmarks latency --target h100/pi0 \
            --plan shipped --plan reference --plan shipped \
            --reps "${TAIL_REPS}" --out "${OUT}/control_pi0.json" > "${OUT}/control_pi0.log"
        sed -n '/^{$/q;p' "${OUT}/control_pi0.log"
        ;;
    experiments)
        echo "== the four treatments, both plans, legs alternating control/treated"
        plan_args=()
        for p in ${TAIL_PLANS}; do plan_args+=(--plan "${p}"); done
        "${PYTHON}" -u -m lab.pi05.tail_experiments --target h100/pi05 "${plan_args[@]}" \
            --reps "${TAIL_REPS}" --legs "${TAIL_LEGS}" --out "${OUT}/experiments.json"
        ;;
    gate)
        echo "== the promotion gate, shipped against reference, baseline tier included"
        "${PYTHON}" -u -m eval.gate --target h100/pi05 --baseline --reps "${TAIL_REPS}" \
            --out-dir "${OUT}/gate" || echo "[job] gate exit $?"
        ;;
    aba)
        echo "== three consecutive A/B/A runs, for the tail bound's repetition requirement"
        for i in 1 2 3; do
            "${PYTHON}" -u -m benchmarks latency --target h100/pi05 \
                --plan reference --plan shipped --plan reference \
                --reps "${TAIL_REPS}" --out "${OUT}/aba_${i}.json" > "${OUT}/aba_${i}.log"
            sed -n '/^{$/q;p' "${OUT}/aba_${i}.log"
        done
        ;;
    *)
        echo "[job] unknown step '${step}'" >&2
        exit 2
        ;;
    esac
done
echo "[job] finished $(date)"
