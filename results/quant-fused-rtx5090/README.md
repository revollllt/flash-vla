# Fused producer-quantize, RTX 5090 (2026-09-23)

What does fusing quantization into the producer of each GEMM A operand save
per observation? The ops are
[`hardware/nvidia/quant_ops`](../../src/flash_vla/hardware/nvidia/quant_ops/README.md);
the script is [`lab/quantization/bench_fused_quant.py`](../../lab/quantization/bench_fused_quant.py);
the tables are in [`summary.md`](summary.md), with raw rows in `raw.json`.
[`compare_flashinfer.md`](compare_flashinfer.md) times the ops against
FlashInfer's own quantize and RMSNorm→NVFP4 kernels
(`lab/quantization/compare_flashinfer.py`).

## Result

Chain time over all producer sites, ms per observation: producer [→ quantize]
→ b12x GEMM(s). `raw.json` / `summary.md` hold the run with PDL, and
`no_pdl/` holds the run with our kernels launched without PDL. FlashInfer's
GEMMs keep their default (PDL) in both.

| model | format | unfused, PDL off | fused, PDL off | unfused, PDL | fused, PDL | fused + PDL saves |
|---|---|---:|---:|---:|---:|---:|
| GR00T N1.7 | MXFP8 | 7.47 | 6.83 | 6.56 | **6.11** | 1.36 ms |
| GR00T N1.7 | NVFP4 | 6.00 | 5.34 | 5.08 | **4.53** | 1.47 ms |
| Pi0.5 | MXFP8 | 17.33 | 16.15 | 16.20 | **15.23** | 2.10 ms |
| Pi0.5 | NVFP4 | 11.11 | 9.67 | 9.92 | **9.06** | 2.05 ms |

These are sums over the listed sites, not end-to-end latency. Each site's chain
is replayed on its own with random data and cold weights. The sums leave out
attention, the output projections and their GEMMs, embeddings, action
projections and host work. For scale: Flash-VLA's shipped BF16 Pi0.5 replays
end to end in 32.2 ms, and the nine Pi0.5 GEMM sites summed here cost
27.5 ms in BF16 cuBLAS under the same conditions. An end-to-end number needs
a quantized Target measured with the latency protocol.

Fusion alone saves 0.64 / 0.66 ms (GR00T) and 1.17 / 1.44 ms (Pi0.5) without
PDL, and 0.45 / 0.55 and 0.98 / 0.86 ms with it. PDL makes every node cheaper,
including the extra quantize node fusion removes, so fusion's share shrinks
while the total drops further. The standalone quantize of attention outputs and
the context costs 0.13–0.22 ms per observation with PDL (0.18–0.27 ms without);
removing it needs an attention epilogue that writes MXFP8 or NVFP4.

The survey's "+ unfused quantize" column charged one FlashInfer quantize per
GEMM call: 0.84 ms (MXFP8) and 0.68 ms (NVFP4) for GR00T. Counting per producer
instead (the refiner's q, k and v share one A, the 16 cross-attention K/V
GEMMs share the context) lowers that. Before PDL, fusion removed 0.61–0.66 ms
in three runs.

## Method

- Sites: every GEMM A operand of GR00T N1.7 and Pi0.5 in the survey shapes,
  with its producer taken from the model references (`models/groot_n17/reference.py`,
  Pi0.5's `torch_ops.py`) and its producers per observation.
- Isolated: the BF16 producer, the standalone quantize (ours and FlashInfer's)
  and the fused producer, each a graph of 64 calls on L2-warm activations.
  Saving = BF16 producer + our quantize − fused producer.
- Chain: producer → quantize → GEMM(s) against fused producer → GEMM(s). The
  GEMMs are FlashInfer b12x on enough cold weight copies to exceed twice the
  L2, as in the survey. Saving = unfused − fused.
- Both measurements capture every variant once and replay them interleaved for
  21 rounds; savings are medians of per-round differences.
- Inputs are random; the NVFP4 global scale comes from the BF16 output. This
  measures performance only; correctness is `tests/test_quant_ops.py` and
  `lab/quantization/check_quant_ops.py`.

## Caveats

- The BF16 producer here is our kernel in BF16 mode, not the shipped Targets'
  producers. When a Target adopts these ops, re-measure against its own route.
- Pi0.5 `llm geglu -> down` (968×16384): the chain saves 31 µs (NVFP4) and
  14 µs (MXFP8) per call where the removed producer and quantize cost 8.6 and
  8.5 µs. The unfused chain writes a 32 MiB BF16 intermediate and reads it
  back, and that traffic likely evicts the GEMM operands from L2. The effect is
  real for this chain but may not transfer to a model. Counting only the
  removed kernels at that site, Pi0.5's fused + PDL saving is 1.99 ms (MXFP8)
  and 1.64 ms (NVFP4) instead of 2.10 and 2.05.
- PDL per call site: Pi0.5 `exp geglu -> down` in NVFP4 is 1.3 µs slower per
  call with PDL (0.23 ms per observation); `pdl=False` there recovers it.
- `compare_flashinfer.md`: ours / FlashInfer, geometric mean. Kernel time
  (CUPTI, PDL off on both sides; a PDL-launched kernel counts its wait as its
  own time): quantize MXFP8 0.80, NVFP4 0.87, RMSNorm→NVFP4 1.11. Graph node
  with PDL on both sides: 0.82, 0.93, 0.87. Without PDL on either: 0.77, 0.84,
  1.08. FlashInfer still wins NVFP4 quantize nodes at M ≥ 512 (up to 1.23×) and
  RMSNorm→NVFP4 at M = 968. Its `rmsnorm_fp4quant` matches our bytes on about
  98% of values and 97–98% of real scale entries; it rounds once where HF Qwen3
  rounds twice. Its `global_scale` is the encode scale, although the docstring
  says the output is divided by it.
- `pdl_sweep/`: the trigger-point sweep behind the PDL default.
