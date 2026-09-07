---
id: technique-row-traversal-fusion
title: "Glue ops are bound by row traversals and launches: fuse to remove a traversal, bracket with PDL"
type: technique
architectures: [sm90]
tags: [kernel-fusion, vectorized-loads, pdl, griddepcontrol, online-softmax]
confidence: measured
reproducibility: snippet
prerequisites: [hw-pdl-gdc]
related: [technique-kernel-fusion, technique-vectorized-loads, technique-pdl-placement, technique-producer-fusion-pdl, pattern-pdl-primary-is-a-resource]
sources: [doc-hardware-unit-test, doc-kernel-design-templates, note-2026-09-06-optimization-campaign-plan]
evidence_basis:
  - evidence_type: benchmark
    source_id: doc-hardware-unit-test
  - evidence_type: upstream-code
    source_id: doc-kernel-design-templates
---

# Glue ops are bound by row traversals and launches

Norm, activation, rope, per-token quantization and softmax are the glue
between the GEMMs, and they are bound by two things that have nothing to do
with arithmetic. **Bytes**: the roofline is the row read plus the row
written, so the only levers are vector width, how many times the row is
traversed, and how much of the pipeline is fused into one traversal; a
separate kernel per op re-reads the row every time. **Launches**: at decode
shapes a row is a few KB, the kernel runs for microseconds, and a launch
costs [launch.lat.dev.ramp], a first-class term on a chain of small ops.
That is why FlashInfer puts `griddepcontrol` in every one of these kernels
rather than only the interesting ones. Method follows FlashInfer's
`norm`, `activation` and `pos_enc` headers and SGLang's per-token
quantization kernels.

Consequences the templates carry:

- RMSNorm reads the row twice, once for the sum of squares and once for the
  scale, and writes it once; that is its roofline, and any change that does
  not reduce the traversal count is noise. Fuse the residual add into it.
- SwiGLU with fused per-token fp8 quantization collapses four traversals to
  one by holding the result in registers across the amax reduction.
- Rope's interleaved pairs are register neighbours; half pairs cost a second
  load, and the layout is buyable offline.
- Row-wise softmax: online versus naive is two traversals against three, a
  performance choice, unlike attention's online softmax, which is forced.
- Accumulate in fp32 always; the upcast rides the load and the sum of
  squares over a few thousand bf16 elements does not survive 8 mantissa
  bits.

## Source-backed fragment

The unit every one of these kernels is written in, from
`elementwise_sm90.cuh` in `doc-kernel-design-templates`:

```cpp
// 16 bytes is the widest load a thread can issue and the width that saturates
// the memory pipe; every one of these kernels is written in units of it.
template <class T>
inline constexpr int vec_size_of = 16 / static_cast<int>(sizeof(T));
```

## Design checks

- The PDL wait sits immediately above the first producer-data read, never
  at the top: a model parameter (the norm weight) was not written by the
  previous kernel and its load should overlap the producer's tail. Holding it
  in registers across the reduction is the trade (`technique-pdl-placement`).
- The trigger is a compile-time knob enumerated over the kernel's phase
  boundaries and swept on the chain, not the kernel.
- A norm that is subtly wrong still trains and still generates text; gate
  parity at full depth, not by eye.
