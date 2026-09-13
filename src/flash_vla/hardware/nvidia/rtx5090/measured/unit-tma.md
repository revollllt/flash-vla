# unit: tma — the copy engine, and the ring that no longer fits

The skill's `tma_ring` probe through `lab/sm120/run_skill_probe.py`.

**Partial by design.** Only sweep A has been run — ring depth against box size,
at 32 CTAs × 1 producer warp, `stride8k` geometry, DRAM regime with L2 flushed.
Nothing here covers CTA scaling, warp scaling, the L2 regime, box geometry or
the saturation frontier, all of which sm90's table does cover. Do not read a
constant here as if it had sm90's validity range.

```bash
CUDA_HOME=... CUTLASS_DIR=... python3 lab/sm120/run_skill_probe.py tma_ring \
  -- --sweeps A --regime dram --json /tmp/tma.json
```

The probe bakes its machine's constants in as module globals — 132 SMs, 227 KB
of shared memory, 3.35 TB/s. The runner replaces them by name and prints every
substitution; without that, the stage × box grid is sized against 227 KB and the
kernel **fails to launch** here rather than being skipped.

## The descriptor is unchanged from Hopper

`cuTensorMapEncodeTiled` legality is identical to sm90: SW32 caps a box row at
32 B, SW64 at 64 B, SW128 at 128 B, unswizzled unbounded; and under SW128 the
row count caps at 256, so `[tma.bytes.txn.max]` is 32 KB exactly as on H100
(`box_dim[1]=256` accepted, 257 rejected). Correctness across contiguous,
2 KB-strided and 8 KB-strided geometries passes with zero coordinate or data
mismatches.

So a tensor-map layout ported from an sm90 kernel stays legal. What changed is
whether the ring built on it fits.

## The measurement

32 CTAs × 1 warp, DRAM, L2 flushed, 268 MB moved per launch:

| stages | box | in-flight | GB/s | ns/txn |
|---:|---:|---:|---:|---:|
| 2 | 2 KB | 128 K | 264.1 | 248.2 |
| 2 | 4 KB | 256 K | 500.1 | 262.1 |
| 2 | 8 KB | 512 K | 862.6 | 303.9 |
| 2 | 16 KB | 1024 K | 1160.6 | 451.8 |
| 2 | 32 KB | 2048 K | **1502.8** | 697.8 |
| 4 | 2 KB | 256 K | 312.6 | **209.6** |
| 4 | 4 KB | 512 K | 622.2 | 210.7 |
| 4 | 8 KB | 1024 K | 1061.0 | 247.1 |
| 4 | 16 KB | 2048 K | **1498.5** | 349.9 |
| 4 | 32 KB | — | *smem 131104 > 101376* | |
| 8 | 2 KB | 512 K | 312.7 | 209.6 |
| 8 | 4 KB | 1024 K | 624.1 | 210.0 |
| 8 | 8 KB | 2048 K | 1234.9 | 212.3 |
| 8 | 16 KB | — | *smem 131136 > 101376* | |
| 16 | 2 KB | 1024 K | 310.3 | 211.2 |
| 16 | 4 KB | 2048 K | 618.8 | 211.8 |
| 16 | 8 KB | — | *smem 131200 > 101376* | |

**`[tma.issue.warp]` — 210 ns per TMA per producer warp.** The floor of the
ns/txn column, reached at depth 4 and unchanged at 8 and 16 (209.6, 209.6,
211.2 — agreeing to 1%). Slightly cheaper than sm90's 248 ns.

**`[tma.stages.warp.knee]` — 4 stages.** Depth 2 does not cover the latency:
248–262 ns at small boxes against depth 4's 210, and far worse at large ones
(698 ns at 32 KB). Depths beyond 4 add nothing. Same knee as sm90.

**`[tma.bw.dev.dram]` — ≤1.50 TB/s at this geometry.** 94% of
`[ld.bw.dev.dram]`'s measured 1598 GB/s plain-load ceiling and 84% of the
datasheet's 1.792. The copy engine reaches essentially what a plain load does,
so it is not itself the constraint.

## The finding that changes kernel design

**`[tma.ring.smem.forbidden]` — sm90's recommended ring does not fit.**

sm90's `[tma.stages.warp.knee]` reads "four stages for cold weights at any box
up to 32 KB". Four stages at 32 KB needs 131104 B of shared memory. This machine
grants 101376 [smem.bytes.cta.max]. It does not run slower; **it fails to
launch.**

What remains reachable at depth 4 tops out at 16 KB boxes (65600 B). By
`[tma.issue.warp]`, halving the box halves that stage's copy column — and the
two rows that *do* reach ~1500 GB/s here (2 stages × 32 KB, 4 stages × 16 KB)
are the only ones that do.

So a TMA pipeline ported from an sm90 kernel has to be re-tiled, not just
recompiled, and the re-tiling is forced by shared memory rather than by the copy
engine.

## What is not established

Everything sm90's table covers beyond sweep A: whether delivered bandwidth
follows the same `CTAs × warps × bytes-per-box` product law, where the
saturation frontier sits, whether there is an anti-scaling dip like sm90's
11.5% at 44–56 CTAs, what the L2-resident regime costs, whether a strided box
is penalised, and whether TMA-load warmth survives intervening traffic. None of
those was run, and none should be assumed from H100.
