#!/bin/bash
#SBATCH --job-name=fixture_check
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:03:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source sbatch/_common.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
require_cuda
"${PYTHON}" -m lab.optimize.fixture_check --out "artifacts/optimization/fixture_check_${SLURM_JOB_ID}.json"
