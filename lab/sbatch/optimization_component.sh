#!/bin/bash
#SBATCH --job-name=opt_component
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source sbatch/_common.sh
require_cuda
report_env
"${PYTHON}" -m lab.optimize run "${EXPERIMENT_DIR}" --until check
