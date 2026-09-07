---
id: technique-register-unpack-sub-byte
title: "sm90 has no sub-8-bit tensor core: unpack INT4 and FP4 weights in registers"
type: technique
architectures: [sm90]
tags: [quantization, fine-grained-quantization, mma-sync, register-fragments, fp8]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-wgmma]
related: [technique-fine-grained-quantization, technique-scale-on-register-fragment, hw-nvfp4, migration-wgmma-to-tcgen05, kernel-fp8-block-scale-gemm]
sources: [doc-cutlass-hopper, doc-kernel-design-templates]
---

# sm90 has no sub-8-bit tensor core: unpack INT4 and FP4 weights in registers

`mma_sm90_gmma.hpp` contains no e2m1 atom, and the narrowest wgmma input is
fp8 (`doc-cutlass-hopper`). Every INT4 and FP4 kernel on H100 therefore
unpacks weights to bf16, fp16, int8 or e4m3 in registers between the shared
tile and the MMA. That unpack sits on the critical path of a
weight-bandwidth-bound kernel, so it is written as bit arithmetic, never as
integer-to-float conversion: a low-precision field dropped into the mantissa
of a fixed exponent yields `2^e + v` for free and one subtract removes the
`2^e`; a float format's fields shift into a wider format's fields with the
exponent-bias difference folded into one multiply; e2m1's eight magnitudes
are exactly one `prmt` source pair, so e2m1 to e4m3 is a register-resident
lookup whose table is built per group by an integer multiply-add that folds
the block scale in. Methods follow Marlin (IST-DASLab) as carried by vLLM and
SGLang, and humming (inclusionAI) for the fused LUT.

Two consequences shape every such kernel:

- **A quantized kernel and its packer are one artifact.** The dequant is
  cheap only because the weights were permuted offline so each thread's 16
  bytes are its MMA operand; a kernel shipped without its matching packer
  produces wrong numbers, not a slowdown.
- **In A16 the scale multiplies the weight before the MMA; in A8 the MMA
  stays in the quantized domain and the scales multiply the accumulator
  once.** Scales on different axes (per row, per column) can only meet on
  the accumulator anyway.

On sm100 and sm120 the block-scaled MMA takes e2m1 and its scales natively;
porting a kernel built on this technique to Blackwell deletes the unpack
rather than translating it.

## Source-backed fragment

The byte permute that makes every 4-bit to 8-bit conversion a table lookup,
from `quant_sm90.cuh` in `doc-kernel-design-templates`:

```cpp
// A selector nibble with bit 3 set switches to sign-replicate mode instead of
// byte select, which is why callers mask selectors with 0x7 and not 0xF.
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t res;
  asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(s));
  return res;
}
```

## Design checks

- At small batch, `mma.sync` over a pre-permuted `cp.async` blob beats wgmma
  over a TMA tile: a permuted weight matrix has no rectangular box for a
  tensor map (`20_marlin_w4a16.cu`).
- fp8 accumulation on Hopper is reduced precision, so an A8 K loop still
  breaks per scale block and promotes on CUDA cores (`11_fp8_two_level_accum.cu`,
  `22_w4a8_gemm.cu`); int8 x int8 to int32 accumulates exactly and does not.
- Spell three-input logic as `lop3` explicitly; ptxas does not always
  contract `(a & b) | c` into one instruction.
