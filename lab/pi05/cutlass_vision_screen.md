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
The copy for cfg0 is timed; no external reset hides its cost. All real-layer
outputs are compared using the existing shallow numerical tolerances.
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

CPU preparation passed syntax and CLI checks. GPU correctness and timings
are pending the exclusive slot. Run from the probe worktree with its existing
`cutlass_gemm_screen.py`, using `--source-checkout` to select the deployed
model and native cache; `--site` accepts `up`, `down`, or `both`.
