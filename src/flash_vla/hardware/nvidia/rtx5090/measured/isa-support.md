# sm_120 — what the instruction set does and does not carry over from sm_90a

The port question for this repository is narrow and answerable: the sm90 tile
library and the kernels above it name about fifty PTX instructions, and each one
either assembles for consumer Blackwell or does not. This file decides every one
of them, says what the "no" rows cost, and names the replacement.

Nothing here is a performance claim. What an instruction *costs* on this machine
is the measured table's job and is not yet measured; see [README](README.md).

## How each row was decided

**`ptxas` is the oracle, not a document.** Every row is a minimal `.entry`
containing one instruction, assembled by `ptxas` from the CUDA 13.1 toolkit
(V13.1.80) against `sm_90a`, `sm_120` and `sm_120a` in turn. A row reads `yes`
when ptxas returns 0 and `no` when it names the target in its own diagnostic.
A case that fails any other way is `BROKEN` — a bug in the probe's PTX, not a
fact about the machine — and the probe refuses to be read until the count is
zero. It is currently 51 cases, 0 BROKEN.

**The manual agrees, and adds one thing ptxas cannot say.** Every family below was
then checked against NVIDIA's *Target ISA Notes* in a local snapshot of the
PTX ISA (9.3) — the per-instruction sections under `ptx-docs/9-instruction-set`.
ptxas decides whether an instruction *assembles*; the manual additionally states
which targets an instruction is *optimized for*, and those are not the same
question. One row below turns on exactly that difference.

| instruction | official Target ISA Notes | sm_120a? |
|---|---|---|
| `wgmma.mma_async` | "Requires `sm_90a`." | no |
| `tcgen05.*` | `sm_100a`, `sm_101a`/`sm_110a`, `sm_100f`/`sm_101f`/`sm_110f` | no |
| `setmaxnreg` | `sm_90a`, `sm_100a`, `sm_101a`/`sm_110a`, **`sm_120a`**, `sm_120f` or higher in family | **yes** |
| `mma` `.kind`/`.block_scale`/`.scale_vec_size` | "requires `sm_120a` and are supported on `sm_120f` or higher in the same family from PTX ISA version 8.8" | **yes** |
| `mma` `.e3m2`/`.e2m3`/`.e2m1` (FP6/FP4) | "requires `sm_120a` and is supported on `sm_120f`" | **yes** |
| `ldmatrix` `.m16n16`/`.m8n16` | `sm_100a`, `sm_101a`/`sm_110a`, **`sm_120a`**, `sm_120f` or higher in family | **yes** |
| `cp.async.bulk.tensor` | "Requires `sm_90` or higher." | yes |
| `barrier.cluster` | "Requires `sm_90` or higher." | yes |
| `mapa` | "Requires `sm_90` or higher." | yes |
| `fence.proxy.async` | "requires `sm_90` or higher." | yes |

The snapshot is PTX ISA 9.3; the ptxas that decided the matrix is CUDA 13.1,
which accepts PTX ISA up to 9.1. They agree on every row checked.

**`sm_120f` is the finding worth acting on.** The manual offers a
*family*-specific target alongside the architecture-specific one for all three
features that need more than plain `sm_120`. `sm_120f` covers GB20x parts in the
same family rather than this one die, so it is the more portable pin unless
something needs `a` specifically.

The probe carries `sm_120f` as its own column, and across all 51 cases it is
**identical to `sm_120a`** — including the three `a`-gated features. For
everything this repository uses, the family target costs nothing and buys
portability across the family, so it is the one to pin.

Two claims in circulation were checked and are **false for this device**, which
is why the probe exists rather than a summary table copied from elsewhere:
sm_120 does *not* keep `wgmma` (a widely-mirrored community wiki says it does),
and sm_120 is *not* limited to cluster size 1 (the same source says it is;
clusters of 8 launch and exchange distributed shared memory here — measured, see
below).

Reproduce with:

```bash
python3 lab/sm120/ptx_support.py
```

## The matrix

```
instruction                         sm_90a    sm_120   sm_120a   sm_120f
------------------------------------------------------------------------
[tma]
tma.load.2d                            yes       yes       yes       yes
tma.load.3d                            yes       yes       yes       yes
tma.load.2d.multicast                  yes       yes       yes       yes
tma.store.2d                           yes       yes       yes       yes
tma.store.3d                           yes       yes       yes       yes
tma.prefetch.2d                        yes       yes       yes       yes
tma.tensormap.prefetch                 yes       yes       yes       yes
tma.bulk.load.raw                      yes       yes       yes       yes
tma.bulk.store.raw                     yes       yes       yes       yes
tma.commit_group                       yes       yes       yes       yes
tma.wait_group                         yes       yes       yes       yes
tma.wait_group.read                    yes       yes       yes       yes
[cpasync]
cpasync.cg.16                          yes       yes       yes       yes
cpasync.commit_group                   yes       yes       yes       yes
cpasync.wait_group                     yes       yes       yes       yes
[mbarrier]
mbarrier.init                          yes       yes       yes       yes
mbarrier.arrive.expect_tx              yes       yes       yes       yes
mbarrier.try_wait.parity               yes       yes       yes       yes
mbarrier.expect_tx.cluster             yes       yes       yes       yes
[wgmma]
wgmma.fence                            yes        no        no        no
wgmma.commit_group                     yes        no        no        no
wgmma.wait_group                       yes        no        no        no
wgmma.mma_async.bf16.n64               yes        no        no        no
wgmma.mma_async.fp8.n64                yes        no        no        no
[mma]
mma.m16n8k16.bf16                      yes       yes       yes       yes
mma.m16n8k16.f16                       yes       yes       yes       yes
mma.m16n8k8.tf32                       yes       yes       yes       yes
mma.m16n8k32.fp8                       yes       yes       yes       yes
mma.m16n8k32.blockscale.mxf8f6f4        no        no       yes       yes
mma.m16n8k64.blockscale.mxf4            no        no       yes       yes
[ldst_matrix]
ldmatrix.x4.b16                        yes       yes       yes       yes
ldmatrix.x4.trans.b16                  yes       yes       yes       yes
stmatrix.x4.b16                        yes       yes       yes       yes
ldmatrix.m16n16.trans.b8                no        no       yes       yes
[cluster]
cluster.mapa                           yes       yes       yes       yes
cluster.barrier.arrive                 yes       yes       yes       yes
cluster.barrier.wait                   yes       yes       yes       yes
cluster.ctarank                        yes       yes       yes       yes
cluster.nctarank                       yes       yes       yes       yes
[warpspec]
setmaxnreg.inc                         yes        no       yes       yes
setmaxnreg.dec                         yes        no       yes       yes
elect.sync                             yes       yes       yes       yes
[tcgen05]
tcgen05.alloc                           no        no        no        no
tcgen05.ld                              no        no        no        no
[misc]
pdl.griddepcontrol.wait                yes       yes       yes       yes
pdl.griddepcontrol.launch              yes       yes       yes       yes
red.global.add.v4.f32                  yes       yes       yes       yes
red.release.gpu.add.u32                yes       yes       yes       yes
fence.proxy.async.shared               yes       yes       yes       yes
fence.proxy.async.global               yes       yes       yes       yes
shfl.sync.bfly                         yes       yes       yes       yes
```

## The one row that forces a rewrite

**`wgmma` is gone, and nothing on this part replaces it.** ptxas, verbatim:

```
Instruction 'wgmma.fence' not supported on .target 'sm_120a'
Instruction 'wgmma.mma_async with floating point types' not supported on .target 'sm_120a'
```

Not "slower", not "emulated" — refused, on both `sm_120` and `sm_120a`. Neither
does the datacenter Blackwell replacement exist here:

```
Instruction 'tcgen05.alloc' not supported on .target 'sm_120a'
```

So consumer Blackwell has **no warpgroup- or CTA-level asynchronous MMA at all**.
The only tensor-core path is warp-level `mma.sync`, which is the instruction
Hopper *also* has and which the sm90 measured table already characterises as the
slower of the two — `[mma.issue.warp]` puts it at a 63% ceiling against wgmma's
95%, and `[mma.xover.n.wgmma]` says it only wins below tile N = 32.

On sm_120 that trade disappears, because there is no other side to trade
against. Every GEMM mainloop in this repository that currently lands on
`wgmma` — `tile/sm90/wgmma.cuh` and its callers in `gemma_expert`,
`gemma_backbone` and `siglip` — needs an `mma.sync` mainloop instead, and the
sm90 constants that shaped those mainloops (`wgmma.issue.wg.ss`'s "N >= 64 or
do not use the tensor core", `wgmma.stages.wg.knee`'s four in flight,
`wgmma.ratio.sm.wg2`'s "one warpgroup saturates") describe an instruction that
is not present and must not be carried across.

What *does* carry across is the feeding machinery: TMA, `mbarrier` with
transaction counts, `cp.async`, `ldmatrix`/`stmatrix`, `elect.sync`,
`fence.proxy.async` and PDL are all `yes`. The producer side of a warp-specialised
pipeline survives the port intact; the consumer side does not.

## The build flag that silently costs three features

Three rows differ between `sm_120` and `sm_120a`:

| instruction | sm_120 | sm_120a |
|---|---|---|
| `setmaxnreg.inc` / `.dec` | `Instruction 'setmaxnreg.inc' not supported on .target 'sm_120'` | accepted |
| `mma...block_scale` (mxf8f6f4, mxf4) | `Instruction 'mma with block scale' not supported on .target 'sm_120'` | accepted |
| `ldmatrix .m16n16 .b8` | `Feature '.m16n16' not supported on .target 'sm_120'` | accepted |

`setmaxnreg` is the one that matters immediately: warp specialisation in this
repository hands a producer warpgroup's registers to the math warpgroups with
it (`gemma_backbone/.../enc_attn.cu` says so explicitly), and it is an
**architecture-conditional** feature. Building for `sm_120` rather than
`sm_120a` does not degrade it — it fails to assemble.

Per the
[Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/),
`a`-suffixed targets are neither forward nor backward compatible, so this is a
deliberate pin, not a default to inherit. The JIT registries and compiler
settings each Target owns (ARCHITECTURE.md: "each Target keeps its own JIT
registry and compiler settings") are where the pin belongs.

A reported toolchain trap worth carrying: `-arch=sm_120a` has been observed not
to propagate the target to ptxas in some toolchain versions, where
`-gencode arch=compute_120a,code=sm_120a` does. The probes here use the
`-gencode` form. Whether CUDA 13.1 still needs it is unverified.

## Clusters and distributed shared memory: supported, and measured

Every cluster instruction assembles for `sm_120`. Assembling is not placing, so
this was run rather than assumed — `lab/sm120/cluster_dsmem.cu` launches a
cluster, has every rank write into rank 0's shared memory through
`map_shared_rank`, barriers, and checks the sum:

```
device: NVIDIA GeForce RTX 5090  cc 12.0  170 SMs
cudaDevAttrClusterLaunch                = 1
cudaOccupancyMaxPotentialClusterSize    = 8
  cluster 1   ok, DSMEM sum = 1  (want 1)  PASS
  cluster 2   ok, DSMEM sum = 3  (want 3)  PASS
  cluster 4   ok, DSMEM sum = 10 (want 10) PASS
  cluster 8   ok, DSMEM sum = 36 (want 36) PASS
  cluster 16  LAUNCH FAILED: a kernel launch error has occurred due to cluster misconfiguration
```

So the DSMEM split-K reduction that ships in `gemma_expert` has a working
substrate here, up to cluster 8. Whether it is *worth* anything is a separate,
unmeasured question: `[cluster.lat.sync]` on sm90 is 0.65 µs at cluster 8 and
that number does not transfer.

**TMA `.multicast::cluster` is the row where "assembles" and "works well" come
apart, and the manual is the only source that says so.** It assembles for
`sm_120` — the instruction itself "Requires `sm_90` or higher" — and the probe
duly reports `yes`. But its Target ISA Notes add:

> `.multicast::cluster` qualifier is optimized for target architecture
> `sm_90a`/`sm_100f`/`sm_100a`/`sm_103f`/`sm_103a`/`sm_110f`/`sm_110a` and **may
> have substantially reduced performance on other targets**, and hence
> `.multicast::cluster` is advised to be used with `.target` `sm_90a` or
> `sm_100f` or `sm_100a` or `sm_103f` or `sm_103a` or `sm_110f` or `sm_110a`.

`sm_120a` is absent from that list. So multicast is available here and NVIDIA
advises against relying on it — which no amount of ptxas probing would have
revealed, and which is probably what the community claim that "cluster-shared TMA
does not exist on SM120" is a garbled version of. It exists; it is deoptimised.

Treat a multicast TMA on this part as a candidate to *measure*, never as a free
fan-out the way an sm_90a design may. How much it actually costs here is
unmeasured.

## Shared memory is the second structural constraint

Not an instruction, but it changes tiling more than most of the rows above.
From the driver on this device, and consistent with the
[Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
("for devices of compute capability 12.0 the maximum shared memory per thread
block is 99 KB"):

| | sm_90a (H100) | sm_120 (RTX 5090) |
|---|---:|---:|
| shared memory per block, opt-in | 227 KB (232448 B) | **99 KB (101376 B)** |
| shared memory per SM | 228 KB (233472 B) | 100 KB (102400 B) |
| max threads per SM | 2048 | **1536** |
| max warps per SM | 64 | **48** |
| registers per SM | 64 K | 64 K |

Measured on the device by walking `cudaFuncSetAttribute`: 99 KB accepted,
100 KB refused with `invalid argument`.

A tile budget inherited from sm90 therefore overshoots shared memory by 2.29x
and occupancy by 1.33x. `[pipeline.stages.wg.knee]`'s "four stages cold, six if
the shared memory is spare" was written against 227 KB; at 99 KB the spare is
gone, and the stage count has to be re-derived rather than reused.

Note also that the tuning guide's "128 KB shared memory capacity per SM" and the
driver's 100 KB are both right and describe different things — 128 KB is the
unified data cache, 100 KB is the maximum shared-memory carveout of it. The
carveout figure is the one a tile budget is built from.

## What this means per unit of the measured table

Every sm90 constant falls into one of three cases. None of them is "reuse".

| sm90 unit | on sm_120 |
|---|---|
| `tma` | instructions all present; **every number must be re-measured** — different DRAM (GDDR7 at 1.792 TB/s against HBM3 at 3.35), different L2 (96 MB against 50), different SM count (170 against 132) |
| `launch` | present; re-measure. `[ld.bw.dev.dram]`'s `1.85 + MB/2.77` is the single most-cited constant in this repository and is an H100 fact |
| `atomic` | instructions all present; re-measure |
| `coop` | present; re-measure |
| `mma` | **half the unit is about an instruction that does not exist.** The `wgmma.*` and `overlap`/`pipeline` constants have no sm_120 counterpart; the `mma.sync` constants have one but must be re-measured, and `[mma.xover.n.wgmma]` becomes meaningless |

The measured table for this machine is therefore created empty rather than
seeded, and `README.md` beside this file lists which probe fills each unit.

## Probe portability

Of the six probes in the `hardware-unit-test` skill, `gmem_atomic` and
`coop_launch` contain no sm90-only construct and should build for `sm_120a` by
passing `arch_flags`; the harness defaults to `-arch=sm_90a` but takes an
override. `tma_ring` names `sm_90a` but uses only instructions that survive.
`mma_rate`, `overlap` and `pipeline_ws` are built on `wgmma` and need an
`mma.sync` counterpart written for this machine — that is new probe work, not a
flag change.
