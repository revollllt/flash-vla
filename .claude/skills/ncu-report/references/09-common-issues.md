# Common Issues & Gotchas

Collected solutions for the recurring frustrations of profiling CUDA kernels — this cluster's first, then the generic ones.

---

## This cluster (acd_u partition, H100, Nsight Compute 2025.4.1)

### `ERR_NVGPUCTRPERM` on a node that runs everything else fine

Counters are permission-gated per node. Only `ACD1-10`, `ACD1-20`, `ACD1-21`, `ACD1-31`, `ACD1-40`, `ACD1-62` allow ncu; submit with `sbatch -w <node>` (pick an idle one from `sinfo -N -p acd_u -n ACD1-10,ACD1-20,ACD1-21,ACD1-31,ACD1-40,ACD1-62 -o "%N %t"`). `sudo` / `modprobe` fixes below are not available to users here.

### `The current user does not have permission to change clocks` / capture aborts before the first pass

Clock locking is denied cluster-wide. Pass `--clock-control=none` to every ncu invocation (the default `base` tries to lock and aborts), never call `pin_gpu_clocks` in an ncu job, and state "unmodified GPU clocks" in the report. Consequence: cross-run deltas under ~5 % are noise — compare structure, not durations.

### torch cannot see the GPU inside the job (`CUDA error 803`, exits 75 in seconds)

Two driver generations share the partition; `sbatch/_common.sh` drops the compat libraries on the new-driver nodes and fails fast otherwise. Resubmit on another ncu-capable node. Do not maintain a node blacklist — the historical one was this bug's shadow.

### The report has several actions and the numbers "don't match the details page"

A parity driver profiles one launch per mode; the details page prints them all in order. Pass `--action N` to every helper and record the index → mode map in `REPORT.md`.

### `source_info(pc)` is None although the wrapper "set -lineinfo"

The hook (`ATTN_NVCC_DEFINES` / `FFN_NVCC_DEFINES` / `ENC_ATTN_NVCC_DEFINES`) must be exported *inside the job*, before the engine builds, and the build cache is keyed on the flag string — a plain `-lineinfo` and a `-lineinfo -DFOO` are different cache entries, so a stale production `.so` cannot be the explanation, but a hook exported on the login node only can. TileLang kernels need the flag through the plan's build options.

### The login node's ncu cannot open the report / `--page details` fails there

The login node carries different Nsight installs than the `cuda/13.1` module. Export `details.txt` / `raw.csv` inside the job (same ncu that wrote the report), and parse with the Python module — `ncu_utils.py` picks the newest install that loads the report; pin one with `NCU_PYTHON_DIR`.

### The upstream helpers say `rule_name` / `estimated_speedup_pct` are missing

On 2025.4.1 the rule dicts carry `rule_identifier`, `rule_message` (`{'title','message'}`) and `speedup_estimation` (`{'type','speedup'}`). `ncu_utils.rule_speedups` accepts both shapes; introspect `sorted(rules[0].keys())` on a new version.

### `pmsampling:*` metrics are absent, yet the details page shows a PM Sampling section

On this version the timelines are named `warpsampling:smsp__pcsamp_warps_issue_stalled_<reason>` and `<UNIT>.TriageCompute.<metric>`; `plot_timeline.py --list` shows what a report holds. A 12–20 µs kernel gets under ten in-kernel samples at the default 1.5 µs interval (`PMSamplingData` rule) — shorten with `--pm-sampling-interval` when the shape matters.

### `sm__ops_path_tensor_op_hmma_*` is 0 on a kernel that clearly runs wgmma

The ops-path counters do not count wgmma on sm90. Read `sm__pipe_tensor_cycles_active`, `sm__pipe_tensor_op_hmma_cycles_active` and `sm__inst_executed_pipe_tensor_op_gmma`.

### The occupancy rules promise 50–90 % on a persistent kernel

They are estimating a design property (1 CTA/SM, big smem ring). Overrule them (playbook pattern O) and diagnose by stall structure and tensor activity.

---

## ncu permissions

### `ERR_NVGPUCTRPERM: The user does not have permission to access NVIDIA GPU Performance Counters on the target device`

Two solutions:

**A) Use sudo (simplest on dedicated servers):**
```bash
sudo ncu [...]
```

**B) Make it persistent (preferred on shared servers):**
```bash
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/ncu.conf'
sudo update-initramfs -u
# reboot, then regular user can run ncu
```

### `Could not deploy stock section files to "/home/USER/Documents/NVIDIA Nsight Compute/..."`

Set `HOME` to a writable directory:
```bash
export HOME=/any/writable/path
ncu [...]
```

This warning is harmless but noisy. ncu falls back to reading from the CUDA install dir.

---

## `-k "regex:..."` matches nothing

1. **Use the demangled name.** Templates produce something like `void my_kernel<(int)8, (int)256>(...)`. Check:
   ```bash
   cuobjdump --dump-function-names ./my_harness
   ```
   Match against the visible string.
2. **Escape regex metacharacters.** `<` and `>` don't need escaping in most regex flavors, but be careful with parentheses.
3. **The kernel might not have been launched.** Run the harness without ncu and confirm the kernel actually runs.

---

## Source view is empty / `action.source_info(pc)` returns None

The binary was compiled without `-lineinfo`. Add it to the nvcc invocation:
```bash
nvcc -O2 -std=c++17 -lineinfo -gencode=... kernel.cu -o harness
```

For JIT / framework-integrated builds:

- **This repo's CUDA backends**: export the `<HOOK>_NVCC_DEFINES="-lineinfo"` hook in the job (`02-harness-guide.md`).
- **TileLang**: pass `-lineinfo` through the plan's build options, or accept PC-level hotspots.
- **TVM-FFI**: hard to inject `-lineinfo`. Easiest fix: build a standalone harness (see `02-harness-guide.md`).
- **PyTorch `torch.utils.cpp_extension.load`**: pass `extra_cuda_cflags=["-lineinfo"]`. But also switch out of `-O3 -G` to avoid heavy debug instrumentation.
- **CUTLASS**: pass `-lineinfo` via `CMAKE_CUDA_FLAGS`.
- **Triton**: harder — Triton's JIT codegen ignores user nvcc flags. To do source-level on Triton, dump the generated PTX, rebuild as a standalone, and profile.

---

## PM sampling returns nothing

1. **You didn't request it.** Add `--section PmSampling --section PmSampling_WarpStates` to the ncu invocation (on 2025.4.1 `--set full` already includes both; the series are then named `warpsampling:*` / `*.TriageCompute.*`, not `pmsampling:*`).
2. **vGPU / MIG environment.** PM sampling isn't supported under virtualization. Use metric-based (non-timeline) analysis only.
3. **Kernel too short.** Kernels under ~20 µs produce few PM samples; what comes back is dominated by warmup noise.
4. **Specific PM metric just isn't available on your GPU / driver / ncu combination.** Some `pmsampling:sm__throughput.*` or `pmsampling:dram__throughput.*` variants may return empty instance arrays even when other `pmsampling:smsp__warps_issue_stalled_*` series work fine on the same report. Always check `m.num_instances() > 0`; if the SM/DRAM timeline is empty, the stall-reason timelines are a reliable proxy.

---

## ncu takes forever to finish

1. **`--set full` needs 45+ replay passes.** That's normal — each pass reruns the kernel for a different metric group. If your kernel takes 3 ms, full profile takes ~15 s; 3 ms → 300 ms kernels are miserable. Mitigation: profile a smaller representative workload if possible.
2. **Kernel launches aren't isolated.** If your binary does other expensive work (data loading, CUDA context init) before every kernel launch, that runs on every replay too. Move it outside the profile window (ncu only profiles kernel launches matching `-k`).
3. **Don't use `-G` (debug).** It regresses performance ~100× and is useless for perf profiling.

---

## Kernel crashes / produces NaN only under ncu

1. **Profiler clock jitter.** Between replays, ncu resets GPU state. Kernels that depend on specific uninitialized values (bad practice) can behave differently. Fix: initialize all inputs.
2. **Out-of-order pre-replay memory**: ncu saves/restores GPU memory between replays but that can expose latent bugs like reading uninitialized global memory.

---

## Metric returns `None`

1. **Wrong metric name.** See `08-sm90-metric-names.md` (H100) or `08-b200-metric-names.md` (B200). Many stock docs use names that exist on neither.
2. **Metric not in the collected sections.** Add the relevant `--section` or `--set`.
3. **Value is legitimately missing.** Some metrics (like tensor pipe counters) return 0 rather than None when the feature wasn't used; but others return None for hardware not present.

Always wrap metric reads in a helper that returns a default:
```python
def safe(action, name, default=None):
    try:
        return action[name].value()
    except Exception:
        return default
```

---

## `ncu_report` import fails

```bash
ls -d /data/apps/cuda/*/nsight-compute-*/extras/python /usr/local/cuda-*/nsight-compute-*/extras/python
export NCU_PYTHON_DIR=/data/apps/cuda/13.1/nsight-compute-2025.4.1/extras/python     # pin for the helpers
.venv/bin/python -c "import sys,os; sys.path.insert(0, os.environ['NCU_PYTHON_DIR']); import ncu_report; print('OK')"
```

If there's still an `ImportError`, check that the module is compatible with your Python version. The `_ncu_report*.so` compiled extension alongside `ncu_report.py` is built for one specific Python version.

---

## TVM-FFI specific

### "I can't profile my kernel, it's built by `tvm_ffi.cpp.build`"

- Locate the cached `.so` and `kernel.cu`:
  ```
  ~/.cache/flashinfer_bench/cache/tvm_ffi/<solution_hash>/
  ```
- You'll see `build.ninja` with the nvcc invocation — note it does *not* include `-lineinfo`.
- **Workaround:** build a standalone harness from the cached `kernel.cu` (see `02-harness-guide.md`).
- **Alternative:** inject `-lineinfo` into `build.ninja` and recompile manually. But this breaks on the next TVM rebuild.

### "The Python benchmarking script runs but ncu sees no matching kernel"

`-k "regex:..."` must match the demangled name. TVM-FFI wraps kernels in `__global__ void kernel(...)` with a fixed name — check with `cuobjdump --dump-function-names <path to .so>`.

---

## PyTorch specific

### Profiling a PyTorch model's kernel

1. Identify the kernel. Use `torch.profiler` to name it, or look for the generated kernel from `torch.compile`.
2. The kernel is often auto-generated (Triton, CUTLASS, cuDNN). For Triton kernels specifically, Triton recompiles between runs and the kernel name changes. Profile a single script invocation.
3. For `torch.compile`-generated Triton, inspect `TORCH_LOGS=+dynamo` or `TORCH_COMPILE_DEBUG=1` to see the emitted code.

### Profiling CUDA Graph-captured kernels

ncu handles CUDA Graph launches fine — each captured kernel shows up as a separate "kernel launch". Use `-k` regex + `-c N` to target.

---

## Reproducibility

### Results jitter between runs

1. **Lock GPU clocks** — not possible here (denied); elsewhere:
   ```bash
   sudo nvidia-smi -lgc <boost_clock_mhz>    # check with nvidia-smi -q -d CLOCK
   # profile
   sudo nvidia-smi -rgc                      # unlock
   ```
2. **Enable persistent mode** (avoids driver unload between invocations):
   ```bash
   sudo nvidia-smi -pm 1
   ```
3. **Pin the CUDA stream** explicitly rather than relying on default stream.

### Reports don't match colleague's results

- Check ncu version (`ncu --version`). Metric names change between major versions.
- Check GPU driver version (`nvidia-smi`). Some metrics only exist on certain drivers.
- Check exact nvcc invocation — a stray `-G` or missing `-lineinfo` makes a big difference.

---

## Output interpretation

### "`sm__throughput = X%`, is that good?"

It depends on the kernel type:
- GEMM / matmul: the tensor pipe should sit near the mma unit's measured ceiling for its tile N (`hardware-unit-test` `[wgmma.*]`); on B200 50%+ SM throughput is the rule of thumb. Below 30% is bad.
- Persistent warp-specialized task loops (this repo): single-digit SM throughput with DRAM at 10–40 % is the *shape* of the kernel, not a verdict — read the stall structure and TMA bytes.
- Element-wise / reduction: usually 10-30%, because they're DRAM-BW-bound.
- Attention / recurrence kernels: varies wildly; compare against a reference implementation.

Always check Speed-of-Light alongside: `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` — against the *measured* streaming ceiling (`[ld.bw.dev.dram]`, `[tma.bw.dev.burst]`), not the datasheet. If DRAM is at the ceiling, low SM throughput is expected and OK. If DRAM is idle AND SM is idle, you're latency-bound.

### "The details page says `Est. Speedup: X%` — is that reliable?"

Yes, mostly. NCU's rule engine does a reasonable job estimating individual rule impact. Caveats:

- The sum of all `Est. Speedup`s is usually > 100%, because rules overlap (fixing A might also help B). Don't add them.
- Rules are per-pattern; the rule engine doesn't know which one is hardest/easiest to fix in your codebase.
- Use `Est. Speedup: X%` to rank patterns by magnitude; use your judgement for ease of implementation.
