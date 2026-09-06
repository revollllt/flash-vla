#!/bin/bash
# Lane C job 1: the Gemma backbone's baselines, before candidate 1 exists.
#
#   MODE=bench sbatch lab/sbatch/gemma_baselines.sh            # any node
#   MODE=ncu   sbatch -w ACD1-20 lab/sbatch/gemma_baselines.sh  # ncu-capable node only
#   MODE=all   sbatch -w ACD1-20 lab/sbatch/gemma_baselines.sh  # both, one node
#
# MODE splits the job because every GPU on all six ncu-capable nodes was
# allocated when this lane started and the baselines must not queue behind a
# profiler capture. The bench numbers are same-node comparable within their own
# job, which is what they are read for; the ncu report is a microarchitectural
# diagnosis and does not enter any latency claim.
#
# Five things, in one job so every number is same-node comparable:
#   1. the shipped and reference routes of both Targets, per call site
#      (`benchmarks kernels --timer cupti`, cold rotating weights)
#   2. cuBLAS at the same shapes (`lab.gemma_backbone.baselines`)
#   3. one fresh `benchmarks profile` on Pi0.5, so the segment's wall time and
#      its attributed kernel time come from ONE run -- the check the campaign
#      asked for on the 620 us inter-kernel gap the floor report implied
#   4. NCU on the incumbent gated FFN: is it stage-limited or box-limited?
#   5. NCU on the incumbent QKV projection, the same question for candidate C2
#
# Clocks are NOT pinned (deployment does not pin them, and ncu refuses the
# permission anyway); latency claims come from a same-process A/B/A elsewhere,
# never from this job.
#SBATCH --job-name=laneC-j1-baselines
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -uo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/third_party/cutlass}"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
export FLASH_VLA_BUILD_VERBOSE="${FLASH_VLA_BUILD_VERBOSE:-1}"

MODE="${MODE:-all}"

WS="${REPO_DIR}/artifacts/ktasks/gemma_backbone"
OUT="${WS}/runs/j1_${SLURM_JOB_ID:-local}"
NCU_DIR="${WS}/profile/j1_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}" "${NCU_DIR}"

require_cuda
report_env
echo "[job] rev=$(git -C "${REPO_DIR}" rev-parse --short HEAD) tree=${REPO_DIR}"
echo "[job] ncu=$(ncu --version 2>/dev/null | tail -1)"
echo "[job] started $(date)"

step() { echo; echo "=================== $* ==================="; }

if [[ "${MODE}" == "bench" || "${MODE}" == "all" ]]; then

# -- 1. the incumbent routes, per call site ---------------------------------
for target in pi05 pi0; do
    for plan in shipped reference; do
        step "kernels ${target} ${plan}"
        "${PYTHON}" -u -m benchmarks kernels --target "h100/${target}" --plan "${plan}" \
            --segment llm_backbone --timer cupti \
            --csv "${OUT}/kernels_${target}_${plan}.csv" \
            2>&1 | tee "${OUT}/kernels_${target}_${plan}.log"
        echo "!! kernels ${target} ${plan} exit ${PIPESTATUS[0]}"
    done
done

# -- 2. cuBLAS at the same shapes -------------------------------------------
step "cublas baselines, both Targets"
"${PYTHON}" -u -m lab.gemma_backbone.baselines --timer cupti \
    --csv "${OUT}/cublas_baselines.csv" 2>&1 | tee "${OUT}/cublas_baselines.log"
echo "!! cublas exit ${PIPESTATUS[0]}"

# -- 3. wall vs attributed kernel time, one run -----------------------------
for target in pi05 pi0; do
    step "profile ${target} shipped (wall vs kernel_time in one run)"
    "${PYTHON}" -u -m benchmarks profile --target "h100/${target}" --plan shipped \
        --out "${OUT}/profile_${target}.json" 2>&1 | tee "${OUT}/profile_${target}.log"
    echo "!! profile ${target} exit ${PIPESTATUS[0]}"
done

fi  # MODE bench

if [[ "${MODE}" == "ncu" || "${MODE}" == "all" ]]; then

# -- 4/5. NCU on the two incumbent kernels this lane wants to replace --------
# The driver issues exactly one call after warmup; both routed wrappers launch
# two kernels (a TileLang RMSNorm and the GEMM), so --launch-count 2 captures
# the pair and nothing else. No --kernel-name filter: the TileLang kernel
# names are generated, and the pair is what a reader has to compare anyway.
ncu_site() {
    local site="$1" tag="$2"
    step "ncu ${tag}"
    ncu --set full --clock-control=none --target-processes=all \
        --launch-skip 0 --launch-count 2 \
        --force-overwrite --export "${NCU_DIR}/${tag}" \
        "${PYTHON}" -u -m lab.gemma_backbone.ncu_driver \
            --target h100/pi05 --plan shipped --segment llm_backbone \
            --site "${site}" --calls 1 \
        2>&1 | tee "${NCU_DIR}/${tag}.capture.log"
    echo "!! ncu ${tag} exit ${PIPESTATUS[0]}"
    if [[ -f "${NCU_DIR}/${tag}.ncu-rep" ]]; then
        ncu -i "${NCU_DIR}/${tag}.ncu-rep" --page details > "${NCU_DIR}/${tag}.details.txt" 2>&1
        ncu -i "${NCU_DIR}/${tag}.ncu-rep" --page raw --csv > "${NCU_DIR}/${tag}.raw.csv" 2>&1
        echo "[ncu] exported ${tag}: $(wc -l < "${NCU_DIR}/${tag}.details.txt") lines of details"
    else
        echo "[ncu] NO REPORT for ${tag}"
    fi
}

ncu_site llm_backbone_norm_gated_ffn gated_ffn_pi05
ncu_site llm_backbone_norm_qkv_rope  qkv_rope_pi05

fi  # MODE ncu

echo
echo "[job] artifacts in ${OUT} and ${NCU_DIR}"
echo "[job] finished $(date)"
