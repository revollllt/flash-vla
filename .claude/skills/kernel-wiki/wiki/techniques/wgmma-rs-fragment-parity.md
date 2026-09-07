---
id: technique-wgmma-rs-fragment-parity
title: "Two A fragments by stage parity, fully unrolled: the fix for ptxas C7518"
type: technique
architectures: [sm90]
tags: [wgmma, register-fragments, loop-unrolling, ldmatrix]
confidence: measured
reproducibility: snippet
prerequisites: [hw-wgmma]
related: [pattern-serialized-wgmma, technique-scale-on-register-fragment, technique-release-on-retirement]
sources: [doc-hardware-unit-test, doc-kernel-design-templates, note-2026-09-06-optimization-campaign-plan, note-2026-09-07-siglip-cuda-backend]
evidence_basis:
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
  - evidence_type: reproduction
    source_id: note-2026-09-07-siglip-cuda-backend
version_sensitive:
  id: vs-cuda-13-ptxas-c7518
---

# Two A fragments by stage parity, fully unrolled: the fix for ptxas C7518

The classic trigger for ptxas C7518 (`wgmma.mma_async instructions are
serialized`) is RS-operand wgmma whose register A fragments are refilled
across a runtime loop that also contains a divergent exit (a watchdog
break): ptxas inserts a `WG.DP` wait before every wgmma. wgmma reads its
register operands asynchronously; refilling the same registers before the
group retires forces ptxas to prove safety with a full wait on every
instruction. Alternating fragments gives the in-flight group exclusive
registers, and unrolling removes the divergent back edge that blocks the
proof. Expect a serialized mainloop to shed a factor of about 2-3 when fixed.

## Fragment

Two A fragments alternated by stage parity plus a fully unrolled stage loop,
with `wait_group<1>` so stage g's retirement frees the fragment stage g+1
overwrites:

```cpp
Tensor tCrA0 = thr.partition_fragment_A(sA);   // stage parity 0
Tensor tCrA1 = thr.partition_fragment_A(sA);   // stage parity 1
// unrolled: (g & 1) ? stage(g, tCrA1) : stage(g, tCrA0)
```

## Design checks

- Ship the pair, fragment alternation and the unroll, rather than either
  half alone.
- SS-operand wgmma (both operands in shared memory) is not exposed to this
  trap; the technique applies to RS form only.
- Read the full ptxas output on every compile and grep for `C7518` and
  `C7515`; the root failure that lets this hide is a build log filtered to
  errors.
