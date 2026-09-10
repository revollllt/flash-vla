# Hopper (H100 / sm90) Kernel Programming Principles — the NCU-facing companion

The sibling of `blackwell-cuda-programming.md` for this repo's actual target. It exists for the same reason the Blackwell one does: an assistant knows the principles but forgets to apply them while writing, and a profiling report is only useful if each abnormal metric maps back to a principle. Use it **before** proposing a kernel design (checklist at the end) and **while** reading a report (each principle names its NCU signals). The diagnosis playbook (`references/06-diagnosis-playbook.md`) cross-references these principles by number.

Machine numbers are cited by their `hardware-unit-test` tag (`python3 .claude/skills/hardware-unit-test/scripts/constants.py --tag <t>`); this document never restates a measured value. The `kernel-design` skill's sm90 wiki (`references/wiki/README.md`) holds the symptom-indexed optimization experience and its templates show how each mechanism is spelled — this document is the bridge from an NCU signal to those.

---

## Target platform

- **GPU:** NVIDIA H100 80GB HBM3 (SXM) — Compute Capability 9.0, 132 SMs (66 TPCs), 50 MB L2, 80 GB HBM3.
- **Toolchain:** Examples use CUDA 13.1 and Nsight Compute 2025.4.1; select a compatible host compiler.
- **Compile target:** `sm_90a` — the `a` suffix unlocks wgmma, TMA and the sm90-only PTX; a plain `sm_90` build silently loses them.

```bash
nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 -lineinfo -o my_kernel my_kernel.cu
```

---

## H100 architecture quick reference

| Parameter | H100 SXM (this host) | Notes for profiling |
|---|---|---|
| SMs | 132 (`device__attribute_multiprocessor_count`) | a persistent grid is 132 CTAs |
| Max warps / SM, per scheduler | 64 / 16 | `smsp__maximum_warps_avg_per_active_cycle` is out of 16 |
| Register file / SM | 64 K × 32-bit; 255 per thread | `setmaxnreg` moves registers between warpgroups |
| Shared memory / SM (configurable) | 228 KB (227 KB per block opt-in) | `launch__shared_mem_config_size` shows the carve-out |
| L2 | 50 MB (`device__attribute_l2_cache_size`) | weights streamed once per layer blow through it by design |
| HBM3 | 80 GB, 3.35 TB/s datasheet | measured cold ceiling: `[ld.bw.dev.dram]`; TMA burst curve: `[tma.bw.dev.burst]` |
| BF16/FP16 tensor (dense) | ~990 TFLOP/s datasheet | measured wgmma clock: `[wgmma.clock.sm]` |
| FP8 tensor (dense) | ~1980 TFLOP/s datasheet | wgmma with `e4m3` / `e5m2` operand types, FP32 accumulate |
| Thread-block cluster | portable 8, non-portable 16 | `[cluster.count.max]`, `[cluster.lat.sync]` |
| Tensor Memory | none (Blackwell only) | accumulators live in registers — the register wall is real |

---

## Hopper-specific features and how they show up in NCU

### 1. Warpgroup MMA (`wgmma`)

Hopper issues tensor-core work per **warpgroup** (4 warps, 128 threads): `wgmma.mma_async` reads A from registers or smem and B from smem through **matrix descriptors**, accumulates into the warpgroup's registers, and is asynchronous — `wgmma.fence` / `wgmma.commit_group` / `wgmma.wait_group N` control completion.

**NCU signals:**
- `sm__pipe_tensor_cycles_active` and `sm__pipe_tensor_op_hmma_cycles_active` — wgmma is counted on the hmma pipe.
- `sm__inst_executed_pipe_tensor_op_gmma` — the wgmma *instruction* share; `smsp__sass_inst_executed_op_shared_gmma.sum` — wgmma reading smem operands.
- Stall `gmma` (aggregate) / `warpgroup_arrive` (per-PC) — warps waiting on group completion or at the fence.
- **Trap:** `sm__ops_path_tensor_op_hmma_*` reads 0 for wgmma work.

**Practice:**
- `wait_group 0` after every issue serializes the pipe; keep ≥ 1 group in flight (`[wgmma.issue.wg.ss]` is the issue-rate constant at N ≥ 64).
- Tile N has a hard efficiency knee; below the crossover (`[mma.xover.n.wgmma]`) use warp-level `mma.sync`.
- One math warpgroup already saturates the pipe at large N; a second one helps only to hide epilogue and issue gaps (ping-pong, FA3 style — wiki `ext-fa3-pingpong`).
- Build flags matter: a stray dependency can serialize the K loop ~3× (wiki `c7518-wgmma-serialization`).

### 2. Tensor Memory Accelerator (TMA)

Bulk tensor copies (`cp.async.bulk.tensor`) driven by a **tensor map** (`CUtensorMap`, built on the host with `cuTensorMapEncodeTiled`), completing on an **mbarrier** with transaction counts (`expect_tx`). One elected thread issues; the whole CTA (or cluster, with multicast) consumes. Also `cp.async.bulk` for 1-D and TMA stores / reductions from smem.

**NCU signals:**
- `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum` — what TMA actually loaded; `..._tma_{st,red}` for stores.
- `sm__pipe_tma_cycles_active`, `smsp__sass_inst_executed_op_tma_*`.
- **The LSU metrics do not see TMA traffic** — `sectors/request` and the coalescing rules score only the scalar side path.
- Consumers waiting on the mbarrier show as `barrier` (CUTLASS `ClusterBarrier::wait`) or `long_scoreboard` (a `try_wait` spin) at the wait line.

**Practice:**
- Size the ring from the unit constants: per-warp issue interval `[tma.issue.warp]`, in-flight bytes per math warpgroup `[wgmma.bytes.wg.tma]`; a ring that only covers L2 latency starves on a cold DRAM stream.
- Box shape and swizzle must match the smem layout the wgmma descriptor expects (wiki `tma-3d-box-row-major`).
- Release ring frames when the consumer *retires* them, not when the next fill begins (wiki `release-on-retirement`).
- A cold weight stream tops out well under the datasheet even with every knob right (wiki `cold-burst-ceiling`).

### 3. Warp specialization and named barriers

Producer warps (TMA issue, scheduling) and consumer warpgroups (wgmma, epilogue) with `setmaxnreg` shifting registers to the math side and named barriers (`bar.sync id, count`) for role-scoped sync. Persistent task loops here run 1 CTA/SM by design.

**NCU signals:**
- Occupancy pinned at ~10 %, registers *and* smem at the limit, grid == 132 — the design, and the reason the occupancy rules must be overruled (playbook O).
- `launch__occupancy_limit_barriers` — sm90 counts named barriers as an occupancy resource.
- Role branches keep `smsp__thread_inst_executed_per_inst_executed.ratio` slightly under 32 — structural.
- `barrier` + `long_scoreboard` at one or two ring-wait PCs — the producer/consumer signature (playbook P).

### 4. Thread-block clusters and distributed shared memory (DSMEM)

CTAs in a cluster co-schedule on one GPC, read each other's smem (`mapa`, `ld.shared::cluster`), multicast TMA loads, and synchronize with `barrier.cluster`.

**NCU signals:** `launch__cluster_size`, `launch__cluster_dim_*`, `launch__occupancy_cluster_pct`, `launch__cluster_max_active`; `smsp__sass_inst_executed_op_dshared_*`; `l1tex__m_*_mem_dshared_op_tma_*`.

**Practice:** every `cluster_sync` costs `[cluster.lat.sync]` and the scheduler fits only `[cluster.count.max]` clusters — put the barrier at the DSMEM reduction, not per K step (wiki `cluster-barrier-placement`); L2 already serves 132 SMs, so a cluster that exists only to share a weight tile must beat it.

### 5. Asynchronous transaction barriers (`mbarrier`)

Arrive / expect-tx / try_wait with parity; the completion mechanism for TMA and the producer/consumer ring. Waits spin in `try_wait` loops with optional `nanosleep`.

**NCU signals:** `barrier` or `long_scoreboard` at the wait line; `sleeping` when backoff is used; `membar` when a fence sits inside the loop.

### 6. Programmatic dependent launch (PDL) and device-scope protocols

`cudaGridDependencySynchronize` / `griddepcontrol.wait` and `launch_dependents` let the next kernel's prologue overlap this kernel's tail; device-scope flag barriers (`red.release` / `ld.acquire` counters) order tasks inside a megakernel.

**NCU signals:** per-launch ramp visible in `FE_B.TriageCompute.gr__ctas_launched_realtime`; counter polls as `long_scoreboard` on `LDG.ACQUIRE` lines, publishes as `membar`.

**Practice:** one release fence per publish; the wait as late as the dependency allows (wiki `prefetch-across-dependency`, `pdl-placement`); a counter hop costs `[atom.lat.dev.hop]`, a launch `[launch.lat.dev.ramp]`, a grid-wide sync `[coop.lat.dev.sync]`; relaunching is only `[coop.ratio.dev.relaunch]` dearer than a grid sync.

### 7. FP8 (E4M3 / E5M2) on the tensor pipe

Doubles wgmma throughput with per-tensor scaling; accumulate in FP32. Appears in NCU as ordinary tensor-pipe activity; the accuracy budget lives in `eval/acceptance.py`, not in the profiler.

---

## What changes versus Blackwell (for readers of the sibling document)

- **No TMEM**: accumulators are registers; register pressure (principle 6) is the binding constraint on a big fused kernel, and `setmaxnreg` is the tool.
- **wgmma, not `tcgen05.mma`**: warpgroup-issued, descriptor-fed from smem, completion by `wait_group`; the `gmma` / `warpgroup_arrive` stall names are the Hopper vocabulary.
- **No CTA pair**: cluster multicast is the operand-sharing mechanism; a single CTA at M = 64 does not leave half the pipe idle the way Blackwell's single-SM MMA does.
- **50 MB L2, not 126 MB**: a per-layer weight stream cannot stay resident; the streaming ceiling, not L2 persistence, is the lever.
- **Same** cluster / DSMEM / TMA / mbarrier / PDL model; the constants differ (cite the tags).

---

## The sixteen principles, with sm90 NCU signals

| # | Principle | Core NCU signals | Typical structural exceptions |
|---|---|---|---|
| 1 | Enough parallelism | `sm__warps_active`, `launch__waves_per_multiprocessor`, `launch__occupancy_limit_*` | persistent task loops (1 CTA/SM by design); decode-shaped work |
| 2 | Coalesced global access | `sectors/request`, `bytes_per_sector`, `UncoalescedGlobalAccess` rule | gather/scatter; **TMA-fed kernels (rules score the side path)** |
| 3 | Reuse through smem / TMA | `lts__t_sector_hit_rate`, `dram__bytes_*`, `..._tma_ld.sum` | element-wise kernels |
| 4 | No bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared*` / wavefronts | broadcast reads; wgmma-swizzled layouts |
| 5 | No divergence | `thread_inst_executed_per_inst_executed.ratio`, branch efficiency | warp-role branches; tails |
| 6 | Register pressure | `launch__registers_per_thread`, `sass__inst_executed_register_spilling_*` | none — spill is a regression |
| 7 | Hide latency (ILP / ring depth) | `long_scoreboard`, `warps_eligible`, issue-active | pointer chasing |
| 8 | Right precision & intrinsics | `sm__pipe_fp64_cycles_active`, `xu` pipe | numerically critical reductions stay FP32 |
| 9 | Minimal host sync | not an NCU metric — nsys via `gpu-profiler-analysis` | control-flow-dependent shapes |
| 10 | Use the tensor pipe (wgmma) | `sm__pipe_tensor_cycles_active`, `inst_executed_pipe_tensor_op_gmma`, `gmma` stall | non-matrix work; below-crossover tiles → `mma.sync` |
| 11 | Grid / block geometry | waves, tail in the timeline, `WorkloadImbalance` | dynamic shapes → persistent + work stealing |
| 12 | Atomic / counter contention | `long_scoreboard` on `ATOM`/`RED`/acquire loads, `membar` | communication kernels |
| 13 | Vectorized / bulk access | LSU pipe %, `mio/lg_throttle`, TMA bytes | unaligned tails |
| 14 | Read-only paths | L1 hit rate, `LDG` vs `LD` | read-write data |
| 15 | Overlap compute and memory | timeline sawtooth, `barrier` at ring waits, `gmma` | compute-bound kernels; < 3 tiles |
| 16 | Sync overhead | `barrier` samples at `BAR.SYNC` / mbarrier waits, `cluster_sync` | correctness-required barriers |

The principle texts themselves (rationale, thresholds, fixes) are the generic ones in the Blackwell sibling's "准则详解" section and need no Hopper rewrite beyond the mechanism substitutions above: `cp.async` → TMA + mbarrier (7, 15), `tcgen05` → wgmma (10), TMEM → registers + `setmaxnreg` (6), CTA pair → cluster multicast (3).

---

## Kernel-type quick reference (what each style violates by design)

| Kernel style | Violates by design | Read instead |
|---|---|---|
| Persistent warp-specialized task loop (this repo's FFN / attention) | 1, 11 (occupancy pinned), 5 (role branches) | stall structure at the ring waits, `gmma`, TMA bytes vs the ceiling |
| Decode-shaped attention (one query) | 1, 11, 15 | split-K over KV; launch-cost arithmetic |
| Element-wise / RMSNorm / RoPE | 3, 10 | fusion into a neighbour's epilogue; PDL chain |
| Reduction / softmax | 5, 1 (late stages), 12 | warp shuffles; own task kind in a task loop |
| GEMM (wgmma) | should violate nothing | tensor pipe vs `[wgmma.*]`, ring depth, epilogue staging (wiki `epilogue-staging-short-k`) |
| Megakernel / interpreter | 16 (device-scope protocols) | counter hops, fence placement, task graph tail (wiki `ext-mpk-megakernel`) |

---

## Checklist before writing or optimizing a sm90 kernel

### Hopper-specific
0. **Compile target** `sm_90a`? (a plain `sm_90` build has no wgmma / TMA.)
0. **Is the matmul on wgmma**, with ≥ 1 group in flight and N above the knee? Below the crossover, `mma.sync`.
0. **Is the stream on TMA**, ring depth from `[tma.*]`, box and swizzle matching the descriptor?
0. **Warp roles**: producer warp(s) with `setmaxnreg.dec`, math warpgroups with `.inc`, named barriers per role?
0. **Cluster only if it pays**: `[cluster.lat.sync]` per sync, `[cluster.count.max]` scheduling — does multicast beat L2?
0. **Protocol placement**: one release per publish, waits as late as possible, PDL trigger position varied and measured.

### Generic
1. Enough CTAs — or a persistent loop whose task list is balanced? (1, 11)
2. Lane ↔ address mapping contiguous on the scalar path? (2)
3. Reuse planned through smem, with the ring sized from constants? (3, 7, 15)
4. Bank conflicts ruled out offline for the smem layout? (4)
5. Branches uniform per warp, apart from role selection? (5)
6. Registers: no spill at 255; `__launch_bounds__` / `setmaxnreg` set? (6)
7. Literals suffixed, fast intrinsics where the accuracy budget allows? (8)
8. Unnecessary block-level syncs replaced by warp-level or mbarrier scopes? (16)
9. Vectorized or bulk access for the side path? (13)
10. `const __restrict__` on read-only pointers? (14)
