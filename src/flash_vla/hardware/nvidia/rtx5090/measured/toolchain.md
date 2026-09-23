# sm_120 — what of the software stack actually runs here

[isa-support.md](isa-support.md) settles what the *hardware* accepts. This file
settles what the *stack above it* does: torch, CUDA graphs, TileLang, CUPTI and
nvcc, each checked by running it on this device rather than by reading a support
matrix. Every row below is a gate that something later depends on, and each one
had a way to come out the other way.

| | |
|---|---|
| device | NVIDIA GeForce RTX 5090, cc 12.0, 170 SMs, driver 580.142 |
| toolkit | `/home/ubuntu/cuda-13.1` — nvcc/ptxas V13.1.80, ncu 2025.4.0, PTX ISA ≤ 9.1 |
| torch | 2.13.0+cu130, `get_arch_list()` includes `sm_120` |
| tilelang | 0.1.11 |

## TileLang works on sm_120, warp specialisation included

This was the go/no-go for the first Target. `hardware/nvidia/h100/pi0/` has **no
hand-written CUDA at all** — its fast path and its `reference` plan are both
TileLang (`pi0/target.py:107`) — so there is no torch fallback to retreat to.

TileLang 0.1.11's arch table knows `sm_120` but not `sm_120a`/`sm_120f`. The
hypothesis was therefore that warp-specialised kernels would fail, because warp
specialisation on Hopper reallocates registers with `setmaxnreg`, which ptxas
refuses on plain `sm_120` [isa.target.a_required]. Three samples, chosen to
separate the failure modes:

| kernel | what it exercises | result |
|---|---|---|
| `tl_rms_norm` | reduction only, no tensor core | PASS, 3.2e-3 rel err vs torch |
| `tl_matmul` | tensor core, warp specialisation **off** | PASS, 2.4e-3 |
| `tl_matmul_ws` | tensor core, warp specialisation **on** | PASS, 2.4e-3 |

**The hypothesis was wrong, and the interesting part is why.** Warp
specialisation is genuinely applied — it is not silently dropped, which would
have made the third row meaningless. The two variants share a builder and differ
only in `TL_DISABLE_WARP_SPECIALIZED`, and their generated CUDA differs
accordingly:

| | source length | `mbarrier` | `__syncthreads` | `ldmatrix` | `setmaxnreg` | `wgmma` |
|---|---:|---:|---:|---:|---:|---:|
| WS off | 8070 | 0 | 2 | 4 | 0 | 0 |
| WS on | 5246 | **15** | 1 | 2 | **0** | 0 |

So TileLang's warp specialisation here is an `mbarrier` producer/consumer
pipeline **without** register reallocation. `setmaxnreg` is a hint in PTX, not a
requirement of the pattern, and TileLang does not emit it on this target — which
is why plain `sm_120` is enough for TileLang and the `a`/`f` target is not.
Neither variant emits `wgmma`, consistent with the ISA matrix.

**What this does not establish.** That these kernels are *fast*. Every shape and
stage count in `pi0/backends/tilelang/wrappers.py` was tuned against 227 KB of
shared memory and 64 warps per SM; this machine has 99 KB and 48
[smem.bytes.cta.max]. Correct and fast are different questions, and only the
first is answered.

## torch and CUDA graphs

The runtime captures and replays a CUDA graph on a dedicated stream
(`runtime/cuda/graph.py`), so graph capture failing would invalidate the whole
execution model, not one kernel.

- bf16 matmul against an fp32 reference: 3.4e-3 max rel err — ordinary bf16.
- CUDA graph capture + replay, then a changed input: bit-identical to eager.

Worth noting from the profile below: torch's bf16 matmul dispatches to
`cutlass_80_wmma_tensorop_bf16_s16816gemm...` — an **sm_80-generation WMMA**
kernel. cuBLAS's own fallback on this part is not a Blackwell-specific kernel,
which is context for whatever "the torch baseline" turns out to cost.

## CUPTI timeline survives the counter-permission block

`ERR_NVGPUCTRPERM` denies NCU and nsys GPU *counters* here. It does **not** deny
CUPTI *activity* tracing: a `torch.profiler` capture with `ProfilerActivity.CUDA`
returns kernel-level device times on this device, naming the kernel. The
top-down model timeline — step 4 of the optimization workflow — is therefore
available; only kernel-counter diagnosis is lost. See
[ncu-metrics.md](ncu-metrics.md).

The counter-free attribution path the repository already owns also assembles
here: `kernel_trace.cuh` reads `%globaltimer` and `%smid`, and both are accepted
for `sm_120f`.

## nvcc compiles sm_90a on this box, which makes the tile refactor verifiable

There is no H100 here, so an sm90 kernel cannot be *run*. It can be *compiled*:

| kernel | sm_90a PTX | `wgmma` instructions |
|---|---:|---:|
| `lingbot_vla/.../split_attention.cu` | 53 951 lines | **0** |
| `lingbot_vla/.../skinny_gemm.cu` | 227 315 lines | **848** |

Two consequences. First, a refactor of the shared tile library can be gated on
**byte-identical sm_90a PTX before and after** — provable behaviour preservation
without the target hardware. Second, the port surface splits cleanly: the
split-key attention kernel is already pure `mma.sync` and should cross with
little more than an arch flag, while the skinny GEMM's 848 `wgmma` instructions
are the real rewrite.

## Machine paths

Every native library resolves its compiler the same way
(`hardware/nvidia/native.py`): `FLASH_VLA_NVCC`, else `CUDA_HOME/bin/nvcc`, else
`nvcc` on `PATH`; LingBot's libraries honour `LINGBOT_NVCC` ahead of that. No
machine path is baked in any more -- the H100 LingBot build used to default to
`/data/apps/cuda/12.6/bin/nvcc`, so a machine that relied on that default now
sets `LINGBOT_NVCC`. The H100 loaders' old per-call `NVCC` variable is gone too;
`FLASH_VLA_NVCC` replaces it. Build the CUTLASS 4.7.1 kernels with a CUDA 13.1 toolkit:
nvcc 13.4 rejects `cutlass_backbone.cu` (2026-09-24, the same source as before
the loader change).
