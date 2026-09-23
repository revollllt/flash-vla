# Quantization

Preliminary home for quantized execution: recipes, weight formats and the
reference quantize/dequantize math. The location may change once the
implementation settles.

**Status (2026-09-23):** no Target runs a quantized policy yet, and
`VLA.precision` is `bf16` everywhere. `formats.py` (block formats, the 128x4
scale layout) and `reference.py` (reference quantize/dequantize and producer
rounding) back the first kernels, the fused producer-quantize ops in
[`hardware/nvidia/quant_ops`](../hardware/nvidia/quant_ops/README.md) (any Blackwell GPU).
The GEMM survey chose MXFP8 over 1D2D block FP8
([results](../../../results/quant-kernel-survey-rtx5090/summary.md)).

## Scope

- **FP8 (E4M3)**, two candidate recipes, both free of calibration:
  - *MXFP8*: weights and activations both grouped [1,32] along K — SFA is
    [M, K/32], SFB is [N, K/32] — with UE8M0 scales applied inside the MMA
    (`kind::mxf8f6f4.block_scale.scale_vec::1X`). It does not go through
    `Sm120BlockwiseScaleConfig`, whose K granularity must equal the tile's K.
  - *1D2D block FP8*: one FP32 scale per [128 N, 128 K] weight block, derived
    from the weights, and one per [1 token, 128 K] activation group, computed
    at run time; two-level accumulation. This is CUTLASS's
    `Sm120BlockwiseScaleConfig<1, 128, 128>` (example 87b).

  The choice is made on fake-quant model quality and on the measured
  throughput of the block-scaled MMA; neither is measured yet.
- **NVFP4** once FP8 is stable: E2M1 values, one UE4M3 scale per 16 and an
  FP32 global scale. The quantized call sites come from a quality-only
  sensitivity sweep; the global activation scale is calibrated.
- **Not now**: weight-only designs such as Marlin (this GPU has native low-bit
  tensor cores), W4A8 and other mixed precision (after FP8 and NVFP4 are stable
  and faster), INT8, quantized attention and KV.

## Decisions the implementation follows

- A recipe fixes formats, scale granularity and source, rounding, scale
  layout and the set of quantized call sites. It is part of the workload's
  execution policy (`Identity.execution_variant.quantization`) and is approved
  by a person on quality evidence: its fake-quant reference, numerical error
  against BF16 and LIBERO task success. Agents optimize kernels inside an
  approved recipe; changing the recipe changes the numerical contract.
- Every recipe has a PyTorch fake-quant reference that implements its exact
  math. Kernels are checked against that reference, not against BF16.
- A quantized workload is a separate comparison context
  ([ARCHITECTURE.md](../../../ARCHITECTURE.md)): its own run directory and
  baseline. BF16 → FP8 is reported as a latency–quality trade-off, never as an
  optimization gain.
- A K dimension that is not a multiple of the scale block is zero-padded, with
  the matching bias entries zeroed (Pi0.5's SigLIP FFN width 4304 is neither a
  multiple of 32 nor of 128).

## Where things live

- This package: model- and hardware-independent pieces — recipe definitions,
  weight formats and scale layouts (`formats.py`), reference quantize/dequantize
  math (`reference.py`). Runtime and Targets may import it; it imports neither.
  Kernels belong under `hardware/`, e.g. `hardware/nvidia/quant_ops`.
- Kernel references: pinned vLLM, SGLang and FlashInfer sources committed in
  `third_party/quant-references/`, CUTLASS examples 79, 87 and 91 in the
  submodule, and kernel-wiki's
  [SM120 source map](../../../.agents/skills/kernel-wiki/references/quantization-sm120.md).
- FlashRT is the comparison baseline and not a reference ([AGENTS.md](../../../AGENTS.md)).
- The design discussion is kept outside the repository for now; it becomes an
  Agent Note under `.agents/notes/proposed/` when implementation starts.
