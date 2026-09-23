# PDL trigger sweep, quant_ops on RTX 5090 (2026-09-23)

Where should `quant_ops` kernels release their programmatic-dependent-launch
dependents? The wait is derived (before the first read of anything an earlier
kernel writes; only the norm weight and bias load above it). The trigger point
changes only speed, so it was swept (kernel-wiki `technique-pdl-placement`):

- point 0: right after the dependency wait
- point 1: once the inputs are consumed, before the stores
- point 2: after the last store (the implicit trigger)

Each configuration ran `lab/quantization/bench_fused_quant.py` on ten sites
(GR00T DiT AdaLN/LN/GELU/attention output, LLM RMSNorm/SwiGLU; Pi0.5 expert
AdaRMS/GeGLU, SigLIP LN, LLM RMSNorm), both formats, in its own process:

```bash
SITES="groot/dit adaln -> qkv,groot/dit ln3,groot/dit gelu,groot/dit attn,groot/llm rms1,groot/llm swiglu,pi05/exp adarms1,pi05/exp geglu,pi05/vit ln1,pi05/llm rms1"
for t in 0 1 2; do
  FLASH_VLA_QUANT_PDL_TRIGGER=$t python lab/quantization/bench_fused_quant.py --only "$SITES" --out pdl_sweep/trigger$t
done
python lab/quantization/bench_fused_quant.py --only "$SITES" --no-pdl --out pdl_sweep/nopdl
```

## Totals over the swept sites, ms per observation

| config | fused producer → GEMM chain | producer → quantize → GEMM chain | fused producer, isolated | standalone quantize, isolated |
|---|---:|---:|---:|---:|
| PDL off | 14.32 | 16.91 | 2.61 | 1.92 |
| **point 0** | **13.88** | **14.84** | **1.94** | **1.35** |
| point 1 | 14.03 | 15.66 | 2.04 | 1.43 |
| point 2 | 14.30 | 16.06 | 2.30 | 1.65 |

Point 0 is the build default (`FVQ_PDL_TRIGGER` in `fused_quant.cu`).

## Pairs that prefer something else

The winning point belongs to the pair (our kernel, the GEMM after it). A
same-process interleaved A/B of PDL on (point 0) against off, fused producer →
b12x chain, two runs agreeing to 0.05 µs:

| site | format | PDL on − off, per link |
|---|---|---:|
| GR00T `dit ln3 -> up` | MXFP8 / NVFP4 | −0.55 / −0.66 µs |
| Pi0.5 `exp adarms1 -> qkv` | MXFP8 / NVFP4 | −0.61 / −0.75 µs |
| Pi0.5 `exp geglu -> down` | MXFP8 | −0.57 µs |
| Pi0.5 `exp geglu -> down` | NVFP4 | **+1.95 µs** (point 1: −0.24 µs) |
| GR00T `llm rms1 -> qkv` | MXFP8 | **+0.51 µs** (point 2: −0.61 µs) |
| GR00T `llm rms1 -> qkv` | NVFP4 | +0.05 µs (point 2: −0.64 µs) |

At those two pairs the early-released b12x GEMM slows the chain. `pdl=False`
on that call recovers all but 0.2–0.6 µs of the best point. A per-call-site
trigger would recover the rest; it is not implemented.

## After the sweep: trigger only in single-wave grids

The sweep's sites all run in one wave (at most 968 CTAs). A later
comparison at larger shapes showed that executing the trigger costs time per
CTA, whether or not the kernel was launched with PDL. The standalone quantize
at 968×16384 (15,488 CTAs of 128 threads) took 15.0 µs with the trigger and 8.3
µs without it, at every trigger point. In a multi-wave grid the trigger cannot
fire early anyway: the last wave starts only as earlier CTAs exit. The kernels
therefore trigger only when the grid fits in one wave (host occupancy query);
other grids leave the release to the implicit trigger at exit. The swept sites
are unchanged by this. In the final full runs
(`../raw.json`, `../no_pdl/raw.json`), the one pair still slower with PDL is
`exp geglu -> down` in NVFP4, by 1.3 µs; `llm rms1 -> qkv` became neutral.
