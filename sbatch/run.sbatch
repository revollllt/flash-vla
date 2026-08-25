#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH -J flash-vla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err

# Generic runner: pass the command through CMD.
#
#   sbatch -J kernelbench --export=ALL,CMD="-m flash_vla.bench --all --timer cupti" sbatch/run.sbatch
#
# CMD is appended to `$PYTHON -u`. Values containing commas break --export
# parsing, so pass those through the environment instead:
#
#   CMD='-m benchmarks profile --stage full --compare' sbatch --export=ALL sbatch/run.sbatch

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"

require_cuda
pin_gpu_clocks
report_env

if [[ -z "${CMD:-}" ]]; then
    echo "[job] CMD is empty; nothing to run" >&2
    exit 2
fi

echo "[job] running: ${PYTHON} -u ${CMD}"
echo "[job] started $(date)"
# shellcheck disable=SC2086
"${PYTHON}" -u ${CMD}
echo "[job] finished $(date)"
