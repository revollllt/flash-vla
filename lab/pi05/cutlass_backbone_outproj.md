# Existing cfg0 on Pi0.5 backbone attention output: rejected

The bounded candidate reused the deployed backbone FFN-down closure and its
cfg0 GEMM (CTA128x128x64, warp64x64x64, 3 stages) for attention out-projection.
Actual x is the contiguous (968,2048) view of attention output (7744,256),
weight is contiguous BF16 (2048,2048), and C=D=out is (968,2048).
The reference calls `out.addmm_(x, weight)`: FP32 accumulation plus FP32 old
out, then one BF16 conversion. There is no intermediate BF16 projection.
The existing native alpha=1, beta=1 path has the same rounding locations.

The 010 backbone trace attributes one GEMM per call and no separate residual
kernel. Its 17 calls total 838.041 us, mean 49.29653 us, median 49.281 us,
range 49.089–49.857 us. The current kernel has 128 CTAs; cfg0 also has 128
output tiles before Stream-K decomposition. A possible scheduling advantage
was the hypothesis, not a claim of eliminated work or fewer launches.

The local run used source revision `cb927cf`, the belt-cup
`orbax-39999+openpi-convert-pi05_aloha` checkpoint, and seed 42. It records
all 17 actual layer activations and initial residuals, retaining the distinct
weights. The graph uses one output buffer per saved call, always as C=D.
All 17 output comparisons have zero maximum absolute difference and zero
relative RMS versus the captured torch route. Stream-K workspace is 131,200
bytes. The 17 weights total 136 MiB, exceeding the 96 MiB L2.

| Route | Total ms / 17 calls | Median us / call |
|---|---:|---:|
| A1 current | 0.895008 | 52.64753 |
| B1 cfg0 | 0.901120 | 53.00706 |
| B2 cfg0 | 0.901120 | 53.00706 |
| A2 current | 0.894976 | 52.64565 |

Each leg retains 15 raw samples. The same residual reset occurs on the capture
stream before each measured graph, excluded from both routes. Control median
drift is 0.000032 ms across 17 calls. The candidate is slower by 0.006128 ms
against the mean control median, so this existing cfg0 reuse is rejected.
No other tile or backbone out-projection implementation was tested or ruled out.
The production wrapper registration was removed; no native/ABI/routing changes
remain. The lab probe calls the existing FFN-down closure directly, reproducing
the same kernel and argument path without registering an out-projection route.

Raw evidence in `/home/ubuntu/flash-vla/artifacts/rtx5090-pi05/`:

- `gpt6-backbone-outproj-local.json`: source, imported plan (`torch` for the control), all 17 numerical checks, and all 60 timing samples.
- `gpt6-backbone-outproj-local.log`: original stdout.

Reproduce with `python -m lab.pi05.cutlass_backbone_outproj --seed 42`, the
converted-checkpoint and checkpoint-identity options, and `--output`.
