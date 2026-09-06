#!/bin/bash
# Lane C: name the gated FFN kernel's epilogue cost, or stop.
#
#   sbatch -w ACD1-20 lab/sbatch/gemma_gu_diag.sh
#
# Where this stands. The mainloop is finished work: copy plus math together are
# 157 us against a 152.85 us [wgmma.clock.sm] floor, 97 % of the tensor core,
# where the incumbent TileLang body sits at 74 %. The epilogue then adds
# ~161 us and the kernel loses 318 to 205 us per call. Two different store
# mechanisms -- a staged shared tile with a TMA store, and bf16 pairs written
# straight from the accumulator -- cost within 10 us of each other, so the
# store mechanism is NOT the cause and the remaining candidates are the
# activation arithmetic or the fact that a persistent CTA's epilogue overlaps
# with nothing.
#
# Six columns, one rebuild each, one job so the rows are same-node comparable:
#
#   copy_only    the ring alone            -> the copy column, never yet measured
#   no_mma       ring + epilogue           -> epilogue over a copy-only baseline
#   no_epilogue  ring + math               -> the mainloop
#   no_gelu      everything but the tanh   -> arithmetic vs stores
#   box3d_full   everything                -> the kernel
#   epi_tma      the staged-store form     -> the other epilogue
#
# Plus an NCU capture of the kernel itself, filtered BY NAME this time: the
# earlier attempt (job 599809) passed --launch-count 2 with no filter, which
# counts from process start and profiled torch's weight-initialisation kernels
# instead. ncu-capable nodes only: ACD1-10/20/21/31/40/62.
#SBATCH --job-name=laneC-gu-diag
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -uo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"
export CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/third_party/cutlass}"
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model}"
export FLASH_VLA_BUILD_VERBOSE=1

WS="${REPO_DIR}/artifacts/ktasks/gemma_backbone"
OUT="${WS}/runs/diag_${SLURM_JOB_ID:-local}"
NCU_DIR="${WS}/profile/diag_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}" "${NCU_DIR}"

require_cuda
report_env
echo "[job] rev=$(git -C "${REPO_DIR}" rev-parse --short HEAD)  ncu=$(ncu --version 2>/dev/null | tail -1)"
echo "[job] started $(date)"

for col in "copy_only:-DGU_ABL_NO_MMA=1 -DGU_ABL_NO_EPILOGUE=1" \
           "no_mma:-DGU_ABL_NO_MMA=1" \
           "no_epilogue:-DGU_ABL_NO_EPILOGUE=1" \
           "no_gelu:-DGU_ABL_NO_GELU=1" \
           "box3d_full:" \
           "epi_tma:-DGU_EPI_TMA=1"; do
    name="${col%%:*}"; flags="${col#*:}"
    echo; echo "=================== ${name} (${flags:-none}) ==================="
    GATED_FFN_NVCC_DEFINES="${flags}" "${PYTHON}" -u -m benchmarks kernels \
        --target h100/pi05 --plan lab/plans/pi05-gemma-cuda-gu.json \
        --site llm_backbone_norm_gated_ffn --timer cudagraph \
        --csv "${OUT}/ablation_${name}.csv" 2>&1 | tee "${OUT}/ablation_${name}.log"
    echo "!! ${name} exit ${PIPESTATUS[0]}"
done

# -- NCU on the kernel itself, and on the incumbent for comparison ----------
ncu_kernel() {
    local regex="$1" tag="$2" plan="$3"
    echo; echo "=================== ncu ${tag} ==================="
    ncu --set full --clock-control=none --target-processes=all \
        --kernel-name-base=function --kernel-name "regex:${regex}" \
        --launch-count 1 --force-overwrite --export "${NCU_DIR}/${tag}" \
        "${PYTHON}" -u -m lab.gemma_backbone.ncu_driver \
            --target h100/pi05 --plan "${plan}" --segment llm_backbone \
            --site llm_backbone_norm_gated_ffn --calls 1 \
        2>&1 | tee "${NCU_DIR}/${tag}.capture.log"
    echo "!! ncu ${tag} exit ${PIPESTATUS[0]}"
    if [[ -f "${NCU_DIR}/${tag}.ncu-rep" ]]; then
        ncu -i "${NCU_DIR}/${tag}.ncu-rep" --page details > "${NCU_DIR}/${tag}.details.txt" 2>&1
        echo "[ncu] ${tag}: $(wc -l < "${NCU_DIR}/${tag}.details.txt") lines"
    else
        echo "[ncu] NO REPORT for ${tag}"
    fi
}

# SKIP_NCU=1 runs the six columns on any node; the capture needs one of
# ACD1-10/20/21/31/40/62 and those were fully allocated for this lane's window.
if [[ -z "${SKIP_NCU:-}" ]]; then
    ncu_kernel "gated_ffn_kernel" mine   lab/plans/pi05-gemma-cuda-gu.json
    ncu_kernel "tl_matmul_gate_kernel" incumbent shipped
else
    echo "[job] SKIP_NCU set: the six ablation columns only"
fi

echo
echo "[job] artifacts in ${OUT} and ${NCU_DIR}"
echo "[job] finished $(date)"
