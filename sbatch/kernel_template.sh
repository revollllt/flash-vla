#!/bin/bash
# Build one kernel-design template on a GPU node and run its built-in harness.
#
#   sbatch --export=ALL,TEMPLATE=45_flag_barrier_megakernel.cu \
#          [,DEFINES="-DFOO=1"][,RUN_ARGS="quick"][,RUN_SEQ="a=1;b=2"][,RUN_TIMEOUT=300] \
#          sbatch/kernel_template.sh
#
# Compiles with the line each template header documents (gencode sm_90a, -O3,
# C++17, CUTLASS on the include path, -lcuda for cuTensorMapEncodeTiled) into
# artifacts/ktasks/templates/runs/<stem>_<jobid>, then runs it. The binary is
# built ON the compute node because the login node has no GPU and a different
# driver; the source is the repo copy, so what runs is exactly what is
# committed. Output lands in sbatch/logs/mkref_<jobid>.out.
#SBATCH --job-name=mkref
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_DIR}"
if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
    . /etc/profile.d/modules.sh
fi
module purge
module load cuda/13.1
module load gcc/13.3
_DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
if [[ -n "${_DRV}" && "${_DRV%%.*}" -ge 580 ]]; then
    LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/compat' | paste -sd: -)"
    export LD_LIBRARY_PATH
fi

TPL_DIR="${REPO_DIR}/.claude/skills/kernel-design/references/templates"
OUT_DIR="${REPO_DIR}/artifacts/ktasks/templates/runs"
mkdir -p "${OUT_DIR}"
: "${TEMPLATE:?set TEMPLATE=<file.cu>}"
STEM="${TEMPLATE%.cu}"
BIN="${OUT_DIR}/${STEM}_${SLURM_JOB_ID:-local}"

echo "[job] node=$(hostname) job=${SLURM_JOB_ID:-local} driver=${_DRV}"
echo "[job] gpu=$(nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader)"
nvidia-smi -lgc "$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader | tr -d ' MHz')" \
    2>/dev/null || echo "[warn] could not lock GPU clocks" >&2
echo "[job] nvcc=$(nvcc --version | tail -1)"
echo "[job] template=${TEMPLATE} defines='${DEFINES:-}' run_args='${RUN_ARGS:-}'"

set -x
nvcc -ccbin "$(command -v g++)" -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 \
     -I"${TPL_DIR}" -I"${REPO_DIR}/third_party/cutlass/include" \
     ${DEFINES:-} -o "${BIN}" "${TPL_DIR}/${TEMPLATE}" -lcuda
set +x
echo "[job] built $(date)"
# RUN_SEQ="a b;c d" runs the binary once per ';'-separated argument set, each
# under a timeout so a deadlocked persistent kernel ends the run, not the job.
if [[ -n "${RUN_SEQ:-}" ]]; then
    # A trailing ';' would be dropped by `read`; use "quick;-" for a quick run
    # followed by a default run ("-" means no arguments).
    IFS=';' read -ra SETS <<< "${RUN_SEQ}"
    for args in "${SETS[@]}"; do
        [[ "${args}" == "-" ]] && args=""
        echo "===== run: ${args}"
        # shellcheck disable=SC2086
        timeout "${RUN_TIMEOUT:-300}" "${BIN}" ${args} || echo "[job] run '${args}' exited $?"
    done
else
    # shellcheck disable=SC2086
    timeout "${RUN_TIMEOUT:-1800}" "${BIN}" ${RUN_ARGS:-} || echo "[job] run exited $?"
fi
echo "[job] finished $(date)"
