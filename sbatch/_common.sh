# Shared environment for every flash-vla GPU job. Source it, do not execute it.
#
# The login node has no GPU and only python 3.6, so everything here targets the
# compute node. flash-vla is not pip-installed anywhere: it runs from source via
# PYTHONPATH, against an interpreter that already has the pinned TileLang.

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/sbatch/logs"

if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
    . /etc/profile.d/modules.sh
fi

module purge
module load cuda/13.1
module load gcc/13.3

# RHEL8 ships GCC 8 as /usr/bin/cc; nvcc -ccbin and torch's JIT must be pointed
# at the module GCC or the TileLang compile fails on ABI grounds.
GCC_BIN="$(command -v gcc)"; GXX_BIN="$(command -v g++)"
export CC="${CC:-${GCC_BIN}}" CXX="${CXX:-${GXX_BIN}}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -ccbin ${GXX_BIN}"

# torch 2.11.0+cu130 + tilelang 0.1.11 -- the combination the recorded baseline
# numbers were measured with. Override with PYTHON=... if that ever changes.
PYTHON="${PYTHON:-/data/user/jzou521/codes/cuda/cuteDSL/.venv/bin/python}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

# TileLang caches compiled kernels against the local device. Keep the cache per
# job so concurrent jobs (and array tasks) never share a half-written entry.
export TILELANG_CACHE_DIR="${REPO_DIR}/.cache/tilelang/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
export TORCH_EXTENSIONS_DIR="${REPO_DIR}/.cache/torch_ext/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${TILELANG_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

pin_gpu_clocks() {
    # Benchmarks are only comparable at a fixed clock. Skipped without permission.
    local gpu_id="${CUDA_VISIBLE_DEVICES:-0}"
    local max_clk
    max_clk=$(nvidia-smi -i "${gpu_id}" --query-gpu=clocks.max.graphics --format=csv,noheader | tr -d ' MHz')
    nvidia-smi -i "${gpu_id}" -lgc "${max_clk}" 2>/dev/null || \
        echo "[warn] could not lock GPU clocks" >&2
}

require_cuda() {
    # Part of the acd_u partition runs a driver this torch build refuses with
    # CUDA error 803, and nvidia-smi still reports a healthy H100, so the job
    # looks fine until torch initializes -- deep inside a benchmark, minutes in.
    # Fail in seconds instead, naming the node, so the retry is obvious.
    if ! "${PYTHON}" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        echo "[job] FATAL: torch cannot see a GPU on $(hostname) (CUDA error 803?)." >&2
        echo "[job] resubmit with --nodelist pinned to a known-good node." >&2
        exit 75
    fi
}

report_env() {
    echo "[job] node=$(hostname) job=${SLURM_JOB_ID:-local} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
    echo "[job] python=$("${PYTHON}" --version 2>&1)"
    echo "[job] torch=$("${PYTHON}" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
}
