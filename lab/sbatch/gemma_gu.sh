#!/bin/bash
# Lane C candidate C1: the persistent warp-specialized gated FFN.
#
#   sbatch lab/sbatch/gemma_gu.sh
#   ABLATE=1 sbatch lab/sbatch/gemma_gu.sh      # also build the ablation columns
#
# Correctness first, then the in-graph subtraction, then the A/B/A. The
# in-graph subtraction is the instrument this lane settled on after C0: two
# isolated timers got the SIGN wrong on a route swap whose operand is
# L2-resident from the kernel before it, and only `benchmarks profile` on both
# plans in one job read it correctly (job 599832).
#
# The ablation columns answer the kernel's own question -- is the mainloop
# bound by its copy or its math -- the way the rejected short-K GEMM note read
# its candidate: rebuild once per column in one job so the rows are same-node
# comparable. GU_BOX_2D is the thesis: the incumbent's 74% of the wgmma
# ceiling should be a 2-D box count, and a 3-D box is one transaction where a
# 2-D tiling needs BK/64.
#SBATCH --job-name=laneC-c1-gu
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

TARGETS="${TARGETS:-pi05 pi0}"
REPS="${REPS:-100}"
ABLATE="${ABLATE:-1}"
WS="${REPO_DIR}/artifacts/ktasks/gemma_backbone"
OUT="${WS}/runs/${TAG:-c1}_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}"

require_cuda
report_env
echo "[job] rev=$(git -C "${REPO_DIR}" rev-parse --short HEAD)"
echo "[job] started $(date)"

step() { echo; echo "=================== $* ==================="; }
run() { local n="$1"; shift; step "${n}"; "$@" 2>&1 | tee "${OUT}/${n}.log"; echo "!! ${n} exit ${PIPESTATUS[0]}"; return 0; }

# -- does it even build and produce the right numbers? ----------------------
for t in ${TARGETS}; do
    run "parity_${t}_gu" "${PYTHON}" -u -m lab.gemma_backbone.parity \
        --target "h100/${t}" --kernel norm_gated_ffn --json "${OUT}/parity_${t}_gu.json"
done

for t in ${TARGETS}; do
    plan="lab/plans/${t}-gemma-cuda-gu.json"
    run "correctness_${t}_l1" "${PYTHON}" -u -m eval.correctness \
        --target "h100/${t}" --plan "${plan}" --steps 1 --layers 1
    run "correctness_${t}_l2" "${PYTHON}" -u -m eval.correctness \
        --target "h100/${t}" --plan "${plan}" --steps 1 --layers 2
    run "correctness_${t}_full" "${PYTHON}" -u -m eval.correctness \
        --target "h100/${t}" --plan "${plan}" --steps 1 --layers 0

    # In-graph, both plans, one job: the only instrument this lane trusts for a
    # route swap. The baseline side is the C0 plan, so the difference is C1.
    run "profile_${t}_base" "${PYTHON}" -u -m benchmarks profile \
        --target "h100/${t}" --plan "lab/plans/${t}-gemma-cuda.json" \
        --out "${OUT}/profile_${t}_base.json"
    run "profile_${t}_gu" "${PYTHON}" -u -m benchmarks profile \
        --target "h100/${t}" --plan "${plan}" --out "${OUT}/profile_${t}_gu.json"

    run "aba_${t}" "${PYTHON}" -u -m benchmarks latency --target "h100/${t}" --reps "${REPS}" \
        --plan "lab/plans/${t}-gemma-cuda.json" --plan "${plan}" \
        --plan "lab/plans/${t}-gemma-cuda.json" --out "${OUT}/aba_${t}.json"
done

# -- the kernel's own decomposition, one rebuild per column ------------------
if [[ "${ABLATE}" == "1" ]]; then
    for col in "box3d_full:" "no_epilogue:-DGU_ABL_NO_EPILOGUE=1" \
               "no_mma:-DGU_ABL_NO_MMA=1" \
               "copy_only:-DGU_ABL_NO_MMA=1 -DGU_ABL_NO_EPILOGUE=1" \
               "no_gelu:-DGU_ABL_NO_GELU=1" \
               "epi_tma:-DGU_EPI_TMA=1"; do
        name="${col%%:*}"; flags="${col#*:}"
        step "ablation ${name} (${flags:-none})"
        GATED_FFN_NVCC_DEFINES="${flags}" "${PYTHON}" -u -m benchmarks kernels \
            --target h100/pi05 --plan lab/plans/pi05-gemma-cuda-gu.json \
            --site llm_backbone_norm_gated_ffn --timer cudagraph \
            --csv "${OUT}/ablation_${name}.csv" 2>&1 | tee "${OUT}/ablation_${name}.log"
        echo "!! ablation ${name} exit ${PIPESTATUS[0]}"
    done
fi

echo
echo "[job] artifacts in ${OUT}"
echo "[job] finished $(date)"
