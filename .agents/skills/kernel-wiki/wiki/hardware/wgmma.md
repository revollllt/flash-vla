---
id: hw-wgmma
title: "Asynchronous warpgroup MMA (wgmma)"
type: hardware
architectures: [sm90, sm90a]
tags: [wgmma, ldmatrix, mma-sync, setmaxnreg]
confidence: measured
reproducibility: snippet
related: [hw-tma, hw-mbarrier, technique-release-on-retirement, technique-wgmma-rs-fragment-parity, technique-scale-on-register-fragment, pattern-wgmma-tile-n-floor, migration-wgmma-to-tcgen05]
sources: [doc-ptx-isa-sm90, doc-hardware-unit-test, doc-kernel-design-templates]
evidence_basis:
  - evidence_type: official-doc
    source_id: doc-ptx-isa-sm90
  - evidence_type: benchmark
    source_id: doc-hardware-unit-test
  - evidence_type: upstream-code
    source_id: doc-kernel-design-templates
aliases: [wgmma, "wgmma.mma_async", WGMMA, "warpgroup MMA"]
---

# Asynchronous warpgroup MMA (wgmma)

`wgmma.mma_async` is Hopper's tensor-core instruction: issued collectively by
a four-warp warpgroup, reading A and B from shared memory through matrix
descriptors (SS form) or A from registers (RS form), accumulating in
registers. Shape `m64nNk16`, N a multiple of 8 up to 256 for bf16. A batch is
`wgmma.fence`, the instructions, `wgmma.commit_group`, then `wgmma.wait_group
N`, which retires all but the newest N groups. Blackwell replaces it with
`tcgen05.mma` (`migration-wgmma-to-tcgen05`).

## Machine facts, by tag

- Issue cost at N=64 or wider, SS operands: [wgmma.issue.wg.ss]; the
  clock the sm sustains under a wgmma stream: [wgmma.clock.sm].
- Below N=64 the instruction is shared-memory-bound, about 3x the issue
  cost; below the crossover [mma.xover.n.wgmma] `mma.sync` wins
  (`pattern-wgmma-tile-n-floor`).
- A second math warpgroup buys no tensor-core throughput
  [wgmma.ratio.sm.wg2]; pingpong is a latency-hiding device
  (`kernel-flash-attention-3`).
- The pipeline knee, groups in flight before retirement stops paying:
  [wgmma.stages.wg.knee]; bytes a warpgroup's wgmma stream can consume from
  TMA: [wgmma.bytes.wg.tma].

## Source-backed fragment

One batch from `03_wgmma_mainloop.cu` in `doc-kernel-design-templates`:
fence the accumulator, arrive, issue every K block, commit, keep one batch in
flight.

```cpp
    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    tiled_mma.accumulate_ =
        (stage == 0) ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
    CUTE_UNROLL
    for (int k = 0; k < size<2>(frag_a); ++k) {
      cute::gemm(tiled_mma, frag_a(_, _, k), frag_b(_, _, k), acc);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    // One batch stays in flight: the previous stage's math overlaps this
    // stage's copies.  warpgroup_wait<0> here would serialize the mainloop.
    warpgroup_wait<1>();
```

## Design checks

- The shared-memory layout and the descriptor's swizzle are one decision:
  the SW128 K-major atom row is 128 B, which is also the widest legal
  swizzled TMA box row.
- RS operands expose ptxas C7518 when register fragments are refilled across
  a loop with a divergent exit (`technique-wgmma-rs-fragment-parity`).
- Release the frame a batch read only after the wait that retires it
  (`technique-release-on-retirement`).
- `setmaxnreg` is `.sync.aligned` over the warpgroup; a lone producer warp
  cannot use it, so make the producer a whole warpgroup and let its unused
  warps release and exit.
