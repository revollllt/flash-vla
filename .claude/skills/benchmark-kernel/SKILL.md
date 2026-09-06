---
name: benchmark-kernel
description: Benchmark a GPU kernel with CUPTI / CUDA-graph / CUDA-event timing, reporting median time, std, achieved TFLOPs and achieved TB/s. Use when measuring per-kernel GPU time for a newly written or newly integrated kernel, comparing one kernel implementation against another, or generating reproducible per-kernel numbers for a PR or a report.
---

# Benchmarking Kernels

Per-kernel GPU timing, ported from FlashInfer's `bench_gpu_time` methodology.
This measures one kernel launch at a time (not the whole pipeline — use
`python -m benchmarks latency` for end-to-end latency).

Three ways in, by how permanent the kernel is:

| | Use when |
|---|---|
| [Method 1: CLI](#method-1-cli-over-a-targets-call-sites) | The kernel is a call site of a Target |
| [Method 2: `bench_gpu_time()` in Python](#method-2-bench_gpu_time-in-python) | You just wrote a kernel and want a number now |
| [Method 3: promote to a call site](#method-3-promote-to-a-call-site) | The kernel is production and needs a tracked baseline |

## Timing Methods

| Method | What it measures | When to use |
|--------|------------------|-------------|
| **CUPTI** (preferred) | Pure GPU kernel execution time (hardware timestamps, no launch overhead) | Always when `cupti-python >= 13` is installed (CUDA 13+); auto-fallback otherwise |
| **CUDA graph** | Launch overhead amortised away, rotating-buffer cold L2 | When CUPTI is unavailable and per-kernel time matters |
| **CUDA events** | Launch + execution (whole wall time of one call) | Coarse checks; launch overhead dominates below ~15 us/call |

The default is CUPTI, falling back to CUDA events automatically when CUPTI is
not installed.

## Method 1: CLI over a Target's call sites

The model is an input. Every call site of the Target's pipeline is a case:
its arguments are recorded from one instrumented eager run of the segment, so
the kernel is timed on exactly the buffers and weights the pipeline passes,
and the recorded invocations are cycled so weight-heavy call sites read cold.

```bash
# Every call site of every segment, amortised in-graph timing (default):
python -m benchmarks kernels --target h100/pi05

# One call site, CUPTI kernel time (auto-fallback to events):
python -m benchmarks kernels --target h100/pi05 --plan reference --segment action_expert --site action_expert_attention --timer cupti

# Pi0's reference route (the fusions off), CSV appended:
python -m benchmarks kernels --target h100/pi0 --plan reference --csv kernel_bench.csv
```

Sibling commands: `python -m benchmarks latency` is the end-to-end latency
baseline, and `python -m benchmarks profile` attributes time *inside* the
captured graph to call sites. `kernels` is the only one that measures a call
site outside any graph, on its own. FLOPs and bytes come from the Target's cost
declarations (`costs.py`), the same minimal-traffic model the floor uses.

On the cluster, run through sbatch — the login node has no GPU:

```bash
sbatch -J kernelbench \
  --export=ALL,CMD="-m benchmarks kernels --target h100/pi05 --plan reference --csv sbatch/logs/kernel_cold.csv" \
  sbatch/run.sbatch
```

### Output

```
action_expert_attention   :: median time 0.021 ms; std 0.001 ms; achieved tflops 16.0 TFLOPs/sec; achieved tb_per_sec 0.33 TB/sec

label                     median ms   std ms   min ms    tflops      TB/s
-------------------------------------------------------------------------
action_expert_attention       0.021    0.001    0.019      16.0      0.33
```

The CSV contains one row per kernel: `label, median_ms, min_ms, mean_ms,
std_ms, p99_ms, tflops, tb_per_sec, num_samples`.

### Cases

A case is `<segment>/<call site>` of the Target on the given plan; the list is
whatever the pipeline calls, so a new call site is benchmarked the moment the
pipeline routes to it. The binding matches production (`op_table` on the same
plan), so the reference route is the default and a candidate plan or table
option selects the alternative.

## Method 2: `bench_gpu_time()` in Python

For a kernel that is not (or not yet) part of flash-vla: any callable that
issues one kernel launch can be measured directly. Nothing about this path is
Pi0-specific — `flash_vla.bench` is only `timer` + `metrics`, ships inside the
installed package, and has no dependency on `benchmarks/` or the Pi0 buffers.
The built-in cases depend on it, not the other way round.

### Step 1: Write your benchmark script

```python
import torch
from flash_vla.bench import bench_gpu_time, KernelResult

# The kernel under test: any callable issuing exactly ONE kernel launch.
def my_kernel(a, b, out):
    torch.matmul(a, b, out=out)      # <- replace with your TileLang/CUDA wrapper

device = torch.device("cuda")
M = N = K = 4096
a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
b = torch.randn(K, N, dtype=torch.bfloat16, device=device)
out = torch.empty(M, N, dtype=torch.bfloat16, device=device)

# Returns the raw per-iteration GPU times in ms (a list, not a summary).
samples = bench_gpu_time(
    my_kernel,
    input_args=(a, b, out),   # pass tensors here, not via closure (see Step 3)
    enable_cupti=True,        # prefer CUPTI, auto-fallback to CUDA events
    repeat_time_ms=100,       # adaptive: iterations sized to ~100 ms of measurement
    dry_run_time_ms=25,       # ...and ~25 ms of warmup
)

# Wrap the samples to get the FlashInfer-style headline metrics.
r = KernelResult("my_kernel", samples, flops=2 * M * N * K,
                 bytes=(M * K + K * N + M * N) * 2)
print(r.perf_line())
print(f"median {r.median_ms:.3f} ms  p99 {r.p99_ms:.3f} ms  n={len(r.samples)}")
```

`flops` and `bytes` are optional — omit them and only the time columns are
reported. For attention-shaped kernels, derive them instead of hand-counting:

```python
from flash_vla.bench import attention_flops, attention_tb_per_sec

flops = attention_flops(batch_size=1, qo_seqlen=51, kv_seqlen=819,
                        head_dim_qk=256, head_dim_vo=256,
                        num_qo_heads=8, causal=False)
tbs = attention_tb_per_sec(batch_size=1, qo_seqlen=51, kv_seqlen=819,
                           head_dim_qk=256, head_dim_vo=256,
                           num_qo_heads=8, num_kv_heads=8,
                           time_ms=r.median_ms)
```

### Step 2: Run it

```bash
python my_benchmark.py                       # a machine with a GPU

# On this cluster, via sbatch (the script must live on shared storage,
# NOT /tmp — compute nodes do not see the login node's /tmp):
sbatch -J mybench --export=ALL,CMD="my_benchmark.py" sbatch/run.sbatch
```

Output of exactly the script above, H100 80GB HBM3, CUPTI available:

```
my_kernel                :: median time 0.179 ms; std 0.001 ms; achieved tflops 766.3 TFLOPs/sec; achieved tb_per_sec 0.56 TB/sec
median 0.179 ms  p99 0.182 ms  n=64
```

`n=64` is the adaptive iteration count: 64 x 0.179 ms lands near the
`repeat_time_ms=100` target.

The same kernel and shapes through the other two backends, for scale:

```
my_kernel[cupti]         :: median time 0.178 ms; std 0.004 ms; achieved tflops 770.7 TFLOPs/sec
my_kernel[cudagraph]     :: median time 0.201 ms; std 0.001 ms; achieved tflops 682.9 TFLOPs/sec
my_kernel[events]        :: median time 0.204 ms; std 0.004 ms; achieved tflops 673.9 TFLOPs/sec
```

The ~13% gap is the CPU-side launch overhead CUDA events include and CUPTI does
not — the reason CUPTI is the default. At 0.18 ms the gap is a rounding error;
at the 5-20 us of a real decoder kernel it is most of the measurement, which is
why per-kernel numbers are only comparable within one backend. When
`cupti-python` is missing the script still runs, emitting
`UserWarning: CUPTI is not installed...` and falling back to the event path
above.

### Step 3: Options that change the measurement

```python
# Force CUDA-graph timing: 10 launches captured per graph, replay time divided,
# so per-call launch overhead is amortised away.
samples = bench_gpu_time(my_kernel, input_args=(a, b, out),
                         enable_cupti=False, use_cuda_graph=True,
                         num_iters_within_graph=10)

# Force CUDA events (launch + execution) even when CUPTI is installed.
samples = bench_gpu_time(my_kernel, input_args=(a, b, out), enable_cupti=False)

# Fixed iteration counts instead of adaptive sizing.
samples = bench_gpu_time(my_kernel, input_args=(a, b, out),
                         dry_run_iters=5, repeat_iters=30)

# Hot L2 (measures the cache-resident best case; off the default path).
samples = bench_gpu_time(my_kernel, input_args=(a, b, out), cold_l2_cache=False)
```

**Pass tensors through `input_args`, not a closure.** A no-argument closure
works, but the harness can only find the tensors it is handed: `cold_l2_cache`
under `use_cuda_graph=True` builds rotating buffer copies from
`input_args`/`input_kwargs` (`calculate_rotation_count` sizes them to exceed
L2), and with nothing to rotate a captured graph re-reads the same
now-L2-resident buffers. Closures still get the L2 flush on the CUPTI and
event paths, where a `zero_()` of a `2x L2` buffer runs between iterations.

**One kernel launch per call.** All three backends attribute one measurement to
one `fn()` invocation. CUPTI correlates the activity records inside the
iteration window and reports `max(end) - min(start)` over them, so a wrapper
that launches two kernels is measured as the *span* covering both, gaps
included — not one kernel, and not the sum. The other two backends measure the
pair silently. The L2 flush is safe here: `zero_()` is issued and synchronised
before the window opens, so its MEMSET record never lands inside it.

### Comparing two implementations

```python
results = []
for label, fn in [("tilelang", tilelang_wrapper), ("torch", torch_reference)]:
    samples = bench_gpu_time(fn, input_args=(a, b, out))
    results.append(KernelResult(label, samples, flops=FLOPS, bytes=BYTES))

from flash_vla.bench import render_table, write_csv
print(render_table(results))
write_csv("compare.csv", results)     # appends; header written on first run
```

## Method 3: Promote to a call site

Once a kernel is production and deserves a tracked baseline, it is a call
site: register its wrapper in the backend's registry, route the pipeline to
it through the op table, and declare its minimal bytes and FLOPs in the
Target's `costs.py`. `python -m benchmarks kernels --target ... --site <name>`
then times it on the arguments the pipeline really passes, `profile`
attributes its in-graph time, and `floor` derives its roofline, with nothing
added to the benchmarks. Nothing in `benchmarks/` names a call site.

The wrapper issues its launches as it does in production (a call site may be
more than one kernel; the profile reports how many); the cost declaration
counts mul+add as two FLOPs and each activation, weight and output once,
matching FlashInfer's `attention_flops` / `attention_tb_per_sec` semantics.

## Troubleshooting

- **`No kernel activities recorded for an iteration`**: the function issued
  extra launches between the CUPTI timestamps (e.g. `buffer.zero_()` for L2
  flush is fine, but a second kernel launch in the same iteration breaks the
  one-iteration-one-kernel correlation). Keep one kernel per `fn()`.
- **`Inconsistent kernel names: ... != ...`**: CUPTI saw a different set of
  kernels on different iterations — usually a wrapper that dispatches by cached
  state, or a JIT that compiles a variant on a later call. Warm the wrapper
  once before benchmarking so every iteration runs the same kernel.
- **`cupti-python` missing / warning at startup**: CUPTI >= 13 requires
  CUDA 13+. Install with `pip install -U cupti-python` in the venv used by
  `sbatch/_common.sh` (`/data/user/jzou521/codes/cuda/cuteDSL/.venv`). Without
  it the benchmark still runs, falling back to CUDA events.
- **CUDA graph capture errors**: a call site must be capturable — no host-side
  allocations on the replay path beyond what graph capture's private pool
  serves. Scratch goes through the engine's `ScratchPool`, which `kernels`
  routes to with the engine's scratch scope.
- **`No such file or directory` on a script you just submitted**: sbatch jobs
  run on a compute node, which does not share the login node's `/tmp`. Keep the
  script under the repo (or any shared path) and pass a repo-relative `CMD`.
- **`CUDA error 803: system has unsupported display driver / cuda driver
  combination`**: the node Slurm picked has a driver too old for the
  `torch 2.11.0+cu130` build in the pinned venv. Not every `acd_u` node has the
  same driver: `ACD1-16` fails this way, while `ACD1-1` and `ACD1-28` run and
  agree to within noise. Resubmit with `sbatch -w ACD1-1 ...` or
  `--exclude=<bad node>`, and always check the `[job] node=` line in the log
  before comparing numbers across runs.
- **Numbers not comparable between runs**: GPU clocks were not locked (this
  cluster denies `nvidia-smi -lgc` without privileges — watch for the
  `[warn] could not lock GPU clocks` line in the job log). Treat cross-run
  deltas > ~5% as suspicious until clocks are pinned.
- **Shapes differ**: a case is timed at the Target's shape profile; the report
  identity names it, and numbers from other shapes are not comparable.

## Reference

- `src/flash_vla/bench/timer.py` — `bench_gpu_time` and the three backends
- `src/flash_vla/bench/metrics.py` — `KernelResult`, `render_table`, `write_csv`,
  `attention_flops`, `attention_tb_per_sec`
- `benchmarks/kernels.py` — `record_invocations` and the `python -m benchmarks kernels` CLI
- Upstream methodology: `flashinfer/testing/utils.py` → `bench_gpu_time` in
  `flashinfer-ai/flashinfer`, and its own `.claude/skills/benchmark-kernel/SKILL.md`
