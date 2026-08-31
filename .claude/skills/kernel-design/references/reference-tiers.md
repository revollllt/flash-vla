# Torch References — tiers, style, numerics

The reference is the oracle: it defines what the kernel must compute, in code a
human can review in minutes. It is never the performance baseline (baselines
are measured implementations) and it never runs under CUDA-graph capture.

## Three tiers, three questions

| Tier | Lives | Answers | Required |
|---|---|---|---|
| T1 algorithm | `src/flash_vla/models/<model>/reference.py` | what the model computes — hardware-free, no padding, no ABI | exists once per model; link it, never fork it |
| T2 ABI mirror | beside the kernel, `backends/cuda/<kernel>_reference.py` | does the kernel compute the right function — same tensors, same buffers, same in-place mutation, deliberately untiled | every kernel task |
| T3 decomposition | same directory | which stage or task inside the kernel is wrong — mirrors the kernel's own split structure | fusion spanning >= 2 pipeline stages |

The simplest possible reference and the ABI mirror pull in opposite directions
(purity vs padding-and-mutation); the tiers exist so neither file does both
jobs badly. Before writing a new reference, open the target's existing
`*_reference.py` files beside its kernels and match their conventions.

## Style rules

- One contraction, one call: `F.linear`, `scaled_dot_product_attention`,
  explicit elementwise ops. No tiling, no cleverness; readability over speed.
- Every tensor edge carries a named-dim comment: `# (M_PAD, D)`. Dim names and
  their fixed values come from ONE geometry-mirror module that derives them
  from the ABI header or the model spec. Two mirrors of one header drift;
  never add a second.
- An ordering that is silent-if-wrong gets a comment naming the constraint
  (e.g. a scale that must multiply before the contraction because it is
  indexed by K and cannot ride the epilogue).
- Weights and factors are inputs; a class exists only to freeze geometry at
  construction. No hidden state.
- Shape asserts at entry are welcome — references never run under capture, so
  they cost nothing that matters.

## Numerics

Mirror the precision placement of the model's official implementation (for
the Pi models: openpi) — do not invent a cleaner one. The pattern: linear
contractions run in the storage dtype (`F.linear` / SDPA on bf16 inputs, with
the fp32 accumulation torch's matmul already performs), while nonlinear ops —
softmax, norms, exponentials, rotations — upcast to fp32 and round back to
the storage dtype at their output. An all-fp32 oracle is wrong on both sides:
it compares the kernel against a function the production model never
computes, and inference tolerates the small error that the mixed placement
admits. Reduction order still differs between implementations, so agreement
is close but never bit-exact — thresholds and where they apply are
`parity.md`'s contract, not this file's.
