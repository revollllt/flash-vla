#!/bin/bash
#SBATCH --job-name=opt_profile
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source sbatch/_common.sh
export PALIGEMMA_TOKENIZER="/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model"
require_cuda
report_env
OUT="artifacts/optimization/profile_${SLURM_JOB_ID}"
mkdir -p "${OUT}"
cp artifacts/optimization/profile-source-state.txt "${OUT}/working-tree.txt"
"${PYTHON}" -m benchmarks profile --target h100/pi05 --plan shipped --steps 1 --layers 1 \
    --trace-dir "${OUT}/trace" --eager-trace-dir "${OUT}/mapping" --out "${OUT}/profile.json"
