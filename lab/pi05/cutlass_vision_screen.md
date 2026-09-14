# Pi0.5 vision FFN CUTLASS screen

Hypothesis: an existing linear Stream-K tile can reduce the deployed vision
FFN GEMMs while preserving bias addition before the BF16 output conversion.
The two actual shapes are M,K,N = 768,1152,4304 and 768,4304,1152.
Each site has 27 layer weights, a 255.34 MiB weight cycle exceeding 96 MiB L2.

The probe records the first actual eager `torch.addmm` operands at each layer
from the selected deployment checkout. It compares the original bias-fused
`torch.addmm` with the existing 12 EpiLinear configurations. Configs 1-11 use
BF16 C=bias, beta=1, ldc=0, giving FP32 accumulator plus bias and one BF16
conversion. GELU and residual addition remain outside this GEMM-only screen.

The deployed config-0 ABI fixes C=D and ldc=N. Its candidate therefore copies
bias into D on every invocation, inside the timed graph, then runs that same
native library with beta=1. This avoids the previously diagnosed GNU-unique
CUTLASS TLS collision between two DSOs with the same config-0 template. A
loss here rejects the current ABI path, not a hypothetical broadcast cfg0.
Each up call writes a 6.3047 MiB expanded bias, and each down call 1.6875 MiB;
the subsequent GEMM logically reads the same C volume. The log records these
volumes per graph. They are logical accesses, not measured DRAM traffic.

Each configuration gets 15 raw CUDA-event samples on a graph of all 27 real
layers, with graph ownership and event timing reused from
`cutlass_gemm_screen.py`. Torch controls bracket each site's configurations.
The copy for cfg0 is timed; no external reset hides its cost. All 27 real-layer outputs are checked individually using the existing
`error_metrics` and shallow tolerances. The summary takes each metric's
worst layer (minimum cosine); all per-layer values remain in the raw JSON.
The output records the imported plan, deployment revision before import,
CUTLASS vendor revision, library paths, geometry, all samples and errors.

## Why this screen follows the backbone measurements

The cold backbone down NCU capture measured 303.168 us, tensor activity
95.8984%, DRAM throughput 20.3593%, L2 throughput 50.2449%, and effective SM
frequency 2.725260 GHz. Its 64.961380352 GFLOP of useful work is 214.2752 TF/s.
At that frequency, the BF16/FP32 instruction limit of 512 FLOP/cycle/SM
(local primitive measurements reached 511.5) across 170 SMs implies
237.2066 TF/s. Useful work reaches
90.3327% of this rate. Accounting for complete 128-row tiles (M padded from
968 to 1024) instead gives 95.5586%, close to the tensor activity counter.
This supports investigating other sites before repeating backbone tile scans;
it does not prove that padding or inactive tensor cycles can be eliminated.
Raw evidence is in the run-01 measurements directory as
`backbone-down-ncu-cold.csv` and `backbone-down-ncu-summary.json`.

Each vision site contains 205.6268 GFLOP across 27 calls. Reusing the down
capture's frequency solely as a scale estimate gives 0.8669 ms of ideal
arithmetic versus the current approximately 1.2-1.4 ms per site. That
0.33-0.53 ms difference per site is not a predicted gain: vision frequency,
Stream-K fixup, edge tiles, epilogue and memory costs have not been measured.
The screen is bounded to these two sites and the existing configurations.

## Measured result, 2026-09-15

Deployment revision before import was `3223a9b`; native CUTLASS used the main
vendor revision `cb4247394dd82148787aed73e5dc7cef33cbf862`. Both sites used all
27 real weights and BF16 biases. All 12 configurations passed every layer.

| Config | Up median (us) | Down median (us) |
|---:|---:|---:|
| 0 | 47.5686 | 46.6477 |
| 1 | 43.3079 | 38.3656 |
| 2 | 43.1858 | 38.6785 |
| 3 | 47.3375 | 41.1378 |
| 4 | 84.2536 | 71.1538 |
| 5 | 41.0335 | 42.9321 |
| 6 | 74.9345 | 67.5129 |
| 7 | 89.6581 | 81.4293 |
| 8 | 45.2741 | 54.2921 |
| 9 | 58.3206 | 61.8844 |
| 10 | 41.5336 | 38.1653 |
| 11 | 52.7846 | 57.3594 |

Config 0 includes its per-call bias copy. All other rows use ldc=0 broadcast.
Torch controls were 45.3547/45.5230 us for up and 47.7357/47.8021 us for down.
Control median drift was 0.1683/0.0664 us; control median absolute deviations
were 0.046-0.066 us for up and 0.105-0.128 us for down. The first sample in a
leg was typically 1-2 us higher, and all raw samples are retained.

Up cfg5 (128x128x32, warp64x64x32, 4 stages) saves 0.117-0.121 ms over 27 calls.
Down cfg10 (64x128x32, warp32x64x32, 5 stages) saves 0.258-0.260 ms. Worst-layer
rel_rms/cosine are 4.914e-5/0.99999999879 and 0.00346349/0.9999940022.
Using only cfg10 for both sites gives approximately 0.36 ms of local savings;
using separate cfg5/cfg10 gives approximately 0.38 ms. The extra template
buys only about 0.014 ms in this screen. No production or E2E comparison has
been performed for either choice.

The initial attempt concatenated all up outputs before `error_metrics` and
hit `torch.quantile`'s input-size limit. It completed the GEMM and failed in
diagnostic statistics. The successful run checked each layer separately;
no numerical criterion changed. Raw artifacts in the deployment checkout's
`artifacts/rtx5090-pi05/` directory are:

- `gpt6-cutlass-vision-screen.json`: samples, per-layer metrics and imported plan.
- `gpt6-cutlass-vision-screen.log`: successful stdout.
- `gpt6-cutlass-vision-screen-first-failed.log`: preserved initial failure.

The script passed syntax/CLI checks and this complete GPU screen. Run it from
the probe worktree with its existing `cutlass_gemm_screen.py`, using
`--source-checkout` for the deployed model/cache and `--site up|down|both`.
