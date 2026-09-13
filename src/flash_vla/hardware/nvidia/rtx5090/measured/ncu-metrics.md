# sm_120 — what Nsight Compute can and cannot see here

The [ncu-report](../../../../../../.agents/skills/ncu-report/SKILL.md) workflow
assumes a metric name resolves. On GB202 many of the names this repository's
saved queries use **do not exist**, and one whole family the sm90 analysis leans
on is gone. This file is the translation table.

Enumerated offline against the installed Nsight Compute (2025.4.0, CUDA 13.1):

```bash
ncu --query-metrics --chip gb202     # 5188 metrics
ncu --query-metrics --chip gh100     # 3691 metrics
```

`--chip` needs no GPU and no permission, so this comparison can be re-run
anywhere the toolkit is installed.

## Counter access: denied to this user, available under sudo

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA
GPU Performance Counters on the target device 0.
```

`NVreg_RestrictProfilingToAdminUsers` defaults to 1 on this host and no
`modprobe.d` entry overrides it, so an unprivileged `ncu` is refused. The name
of that parameter is also the workaround: **root is an admin user**, so running
the profiler under `sudo` works with no module change and no reboot. Verified
here -- `--query-metrics` returns 7987 metrics against the live device, and a
real capture collects counter values.

`sudo` resets `PATH` through `secure_path` and drops most of the environment,
which matters because TileLang shells out to `nvcc` during JIT. Pass what the
run needs explicitly:

```bash
sudo env HOME=$HOME CUDA_HOME=/path/to/cuda PATH=/path/to/cuda/bin:$PATH \
  /opt/nvidia/nsight-compute/2026.2.1/ncu --launch-count 1 --metrics <list> \
  /path/to/.venv/bin/python -m <module>
sudo chown $USER:$USER <report>.ncu-rep
```

Use `/opt/nvidia/nsight-compute/2026.2.1/ncu` rather than the one bundled with
CUDA 13.1 (2025.4.0); it is newer and matches the documentation snapshot.

Setting the module parameter instead would let an unprivileged `ncu` work, at
the cost of making GPU performance counters readable by every local user. On a
single-user box that is a fair trade; `sudo` per capture avoids the question.

**The timeline never needed either.** `ERR_NVGPUCTRPERM` gates *counter*
collection. CUPTI *activity* tracing -- kernel start/end timestamps -- is a
separate mechanism and needs no permission, which is what
`tools/profiling/model.py:151` uses (`torch.profiler` with
`ProfilerActivity.CUDA`). Measured here: a capture returns kernel-level device
time, naming the kernel (`cutlass_80_wmma_tensorop_bf16_s16816...` for a bf16
matmul). So step 4 of
[the optimization workflow](../../../../../../docs/optimization.md) runs
unprivileged, and `sudo` is only needed once the timeline has selected a kernel
worth NCU's attention.

## The tensor-core metric family was renamed, and lost its instruction breakdown

This is the significant one. On GH100 there are **26** `pipe_tensor_op_*`
metrics; on GB202 there are **0**.

| GH100 (sm90) | GB202 (sm_120) |
|---|---|
| `sm__inst_executed_pipe_tensor_op_hmma` | `sm__inst_executed_pipe_tensor_subpipe_hmma_op_hmma` |
| `sm__inst_executed_pipe_tensor_op_imma` | `sm__inst_executed_pipe_tensor_subpipe_imma_op_imma` |
| `sm__inst_executed_pipe_tensor_op_gmma` | **no counterpart** |
| `sm__inst_executed_pipe_tensor_op_dmma` | **no counterpart** |
| `sm__pipe_tensor_op_hmma_cycles_active` | `sm__pipe_tensor_cycles_active` (aggregate only) |

`op_gmma` is the warpgroup-MMA counter. Its absence is the counter-side
confirmation of what ptxas already refused: there is no `wgmma` on this part, so
there is nothing to count. `op_dmma` is the FP64 tensor path, also absent.

Any saved query or playbook step that names `sm__inst_executed_pipe_tensor_op_*`
has to be rewritten against `subpipe_*` before it resolves here.

**GB202 gains something better for this repository's bf16 work**, though: a
direct operation count by source and destination type, which GH100 does not
expose in this form —

```
sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32
sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off
```

Every tensor-core mainloop in this project is bf16 in, fp32 accumulate. That
metric counts exactly that path, and the `sparsity_off` variant removes the
usual ambiguity about whether a quoted peak assumed 2:4 sparsity. It is the
right denominator for a tensor-utilisation claim on this machine.

## DRAM counters were renamed

| GH100 | GB202 |
|---|---|
| `dram__bytes_read.sum` | `dram__bytes_op_read.sum` |
| `dram__bytes_write.sum` | `dram__bytes_op_write.sum` |
| `dram__sectors_read.sum` | `dram__sectors_op_read.sum` |
| `dram__sectors_write.sum` | `dram__sectors_op_write.sum` |

`dram__bytes`, `dram__throughput`, `gpu__time_duration`, `sm__cycles_active`,
`sm__throughput`, `l1tex__t_sector_hit_rate`, `lts__t_sector_hit_rate` and
`smsp__inst_executed` all resolve unchanged.

The rename is silent in the worst way: a query asking for `dram__bytes_read.sum`
does not return zero, it fails to resolve, and a harness that tolerates a
missing metric will quietly report on fewer counters than it was asked for.

## TMA and distributed shared memory are both visible

50 metrics match `tma` on GB202 against 58 on GH100, and the ones the sm90
analysis uses for copy-engine accounting
(`l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld`,
`l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_*`) are present.

77 metrics match `mem_dshared` — distributed shared memory is a first-class,
separately-counted traffic class here, including its TMA forms
(`l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_tma_st` and `_red`). That
corroborates the cluster probe from the counter side: DSMEM is not a vestigial
path on consumer Blackwell.

## What to capture first, once permission exists

The two units that block design work are `launch` and `mma`
([README](README.md)). In counter terms:

| question | metrics (all verified present on `gb202`) |
|---|---|
| is a kernel launch-bound rather than bandwidth-bound? | `gpu__time_duration.sum`, `sm__cycles_active.avg`, `sm__cycles_elapsed.avg` |
| what fraction of the tensor core is a warp-level `mma.sync` mainloop reaching? | `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum`, `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` |
| is the copy pipeline or the math the constraint? | `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum`, `dram__bytes_op_read.sum`, `dram__throughput.avg.pct_of_peak_sustained_elapsed` |
| did the 99 KB shared-memory ceiling cost occupancy? | `sm__warps_active.avg.pct_of_peak_sustained_active`, `sm__warps_launched.sum` |

The last row is the one with no sm90 precedent worth reusing: at 99 KB per block
and 48 warps per SM, the occupancy arithmetic that shaped the sm90 kernels does
not carry over, and it is cheap to check directly rather than re-derive.

**On `launch__*`, corrected against the official documentation.**
`--query-metrics` lists hardware counters only, so `launch__*` names appear on
neither `gb202` nor `gh100` in that listing. That is a property of the listing,
not of the chip. Nsight Compute's own metrics reference documents them:

> `launch__*` metrics are collected per kernel launch, and do not require an
> additional replay pass. They are available as part of the kernel launch
> parameters (such as grid size, block size, ...) or are computed using the CUDA
> Occupancy Calculator.

`launch__grid_size`, `launch__waves_per_multiprocessor`,
`launch__occupancy_per_shared_mem_size` and `launch__occupancy_limit_shared_mem`
are all documented, so use them. Three more are worth knowing here because this
machine's cluster behaviour had to be established by running a probe:
`launch__cluster_size`, `launch__cluster_max_active` and
`launch__cluster_max_potential_size` report from the launch what
`cluster_dsmem.cu` had to discover by trial.

"Do not require an additional replay pass" also suggests these may survive the
counter-permission block above, since they are read from launch parameters
rather than collected from counters. Not tested — NCU may still refuse to attach
at all.
