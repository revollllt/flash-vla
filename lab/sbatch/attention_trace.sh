#!/bin/bash
#SBATCH --job-name=attn_trace
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source sbatch/_common.sh
export PALIGEMMA_TOKENIZER=/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
require_cuda
report_env
BASE="${REPO_DIR}/artifacts/optimization/async-attention-001"
"${PYTHON}" -u -m lab.optimize.attention_trace_probe --inputs "${BASE}/inputs" \
    --traced-inputs "${BASE}/traced-inputs" --out "${BASE}/run_${SLURM_JOB_ID}" \
    --cutlass "${REPO_DIR}/third_party/cutlass"
cuobjdump --dump-resource-usage "${BASE}/run_${SLURM_JOB_ID}/off/libgemma_enc_attn.so" > "${BASE}/run_${SLURM_JOB_ID}/off-resources.txt"
cuobjdump --dump-resource-usage "${BASE}/run_${SLURM_JOB_ID}/traced/libgemma_enc_attn.so" > "${BASE}/run_${SLURM_JOB_ID}/trace-resources.txt"
