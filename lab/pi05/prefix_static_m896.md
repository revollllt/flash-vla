# Static M896 cfg0 screen for prefix QKV and output projection

This lab-only probe tests one fixed configuration on two remaining dense sites
after deployed 023. It changes no production code, route, CUDA source or vendor
code. The new information is static M896 at the actual 895-valid-row latency
input, not another M968 cfg0 comparison.

Earlier evidence already bounds expectations:

- M968 QKV cfg0 was numerically exact but its apparent 0.000864 ms / 18-call
  gain was below 0.008768 ms control drift
  (`results/rtx5090-pi05/gpt6-prefix-cfg0/local.json`).
- M968 output projection cfg0 was exact but about 0.006128 ms / 17 calls slower
  than torch (`results/pi05-rtx5090/gpt6-run-01/measurements/backbone-outproj-cfg0-rejected.json`).
- Seed42 recorded runtime masks have 127 prompt tokens / 895 valid prefix rows
  (`results/rtx5090-pi05/gpt6-padding-cpu/padding_runtime_rows.json`).
  896 is an M128 tile boundary, not a task-length constant.

`cutlass_backbone._Plan` already derives M from the passed activation view and
binds the same cfg0 at M896. Slicing contiguous rows preserves A/B data pointers
and leading dimensions. The beta=1 epilogue has C=D; it keeps FP32 accumulation
and residual addition followed by BF16 output. QKV still materializes its BF16
projection before the original adjacent-pair RoPE. No approximation is added.

Removing one of eight M128 row tiles is an arithmetic-work opportunity. It does
not predict speed: Stream-K mapping and grid size change, and current Torch
GEMMs use a different tile. Earlier totals suggest a proportional opportunity
on the order of 0.1 ms per site, before overhead and efficiency changes. Only
the same-condition ABBA will establish whether a gain exceeds drift.

## Fixed experiment and source reuse

`prefix_static_m896.py` builds the current shipped Target once, calls actual
`engine.sample_inputs(42)`, and records both sites in one unchanged prefix
traversal. It saves each of 18 QKV layer inputs/weights/RoPE/expected outputs and
each of 17 out-projection inputs/weights/initial residuals/expected outputs.
Weights come from the existing converted belt-cup checkpoint path in the
command below. No synthetic matrix or reconstructed CPU RNG input is used.
The loaded runtime revision, actual engine identity/plan, probe revision,
source path and library path are stored separately.

The expected n_valid=895 and device mask are checked at recording time. These
are controls for this particular static experiment, not production invariants.

QKV A is the current fused-prefix-qkv wrapper with full968 RMSNorm and original
torch.mm/full968 RoPE-scatter. B retains both full968 pointwise launches and
uses only a static M896 GEMM. A/B share normed, BF16 projected, Q/K/V working
addresses and every immutable input/weight. Both receive one projected.zero_
before each timed graph, outside its timing. B's uncomputed projected tail is
therefore zero and full968 scatter produces finite Q/K/V there. This explicit
reset is part of the local screening conditions; a production candidate would
need its own finite-tail handling and runtime selector.

Output projection A is the actual shipped torch wrapper. B is static cfg0
M896 with alpha=1,beta=1,C=D. A/B share all inputs and one working output address.
Every call first restores its own full968 initial residual **inside** the timed
chain, identically for A and B. B leaves the last72 restored residual rows
untouched. This boundary differs from the earlier M968 rejected probe, whose
per-layer output resets were outside timing; compare the new A/B directly,
not their absolute times with that older artifact.

All A outputs and full968 x_norm are compared with actual captured values.
B Q/K/V and out are compared on both the first895 valid rows and all896
computed rows; padded outputs must be finite across their full physical968
shape. Exact outputs bypass unnecessary error quantile work. Non-exact outputs
use existing shallow tolerances and error_metrics. Any failure is saved and
raised before timing that site.

Each site receives exactly one A1/B1/B2/A2, 15 raw samples per leg, no tile sweep.
All layer weights rotate (180 MiB QKV / 136 MiB output projection), no L2 flush.
There is no claim that this reproduces complete-model cache/traffic conditions.
A negative or drift-scale result ends that site's static candidate.

## Command after exclusive GPU authorization

Run the new script from its isolated worktree, but import production from the
already compiled main checkout. Current main 7869a5a has the 023 libraries warmed
by its correctness, E2E and profile runs. The existing loaders then reuse those
same library paths; do not point production imports to a new worktree cache.

~~~sh
cd /home/ubuntu/flash-vla
source artifacts/rtx5090-pi05/gpt6-env.sh
export CUTLASS_DIR=/home/ubuntu/flash-vla/third_party/cutlass
/home/ubuntu/flash-vla/.venv/bin/python \
  /home/ubuntu/flash-vla-gpt6-backbone/lab/pi05/prefix_static_m896.py --seed 42 \
  --option converted_checkpoint=/home/ubuntu/models/pi05_belt_cup_pytorch \
  --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --output /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-prefix-projections-static-m896.json
~~~

The only lab dependency is existing `lab.pi05.cutlass_gemm_screen.samples_ms`
already present in main. Native code is reused from the current Target.
Preparation validation is Python py_compile and a scoped diff check only;
no Torch/CUDA/model/JIT/NVCC execution has occurred for this probe.

## Measured decisions

Runtime source and loaded identity: `754d7f43d975a86c9488dd2218d2c0a6987183e0`
(the retained 023 production implementation). Probe revision:
`32ea1aac164a4873d509bbffc19a9bf962a93dbb`. Recording confirmed 895 valid rows
and mask[896] = -3.00405527047391e38. All native calls used the existing main
library; the original log has zero ptxas lines.

| Site, total ms | A1 Torch968 | B1 cfg0 M896 | B2 cfg0 M896 | A2 Torch968 |
| --- | ---: | ---: | ---: | ---: |
| QKV full chain, 18 calls | 1.044480 | 1.042432 | 1.042432 | 1.043808 |
| Out-projection with reset, 17 calls | 0.912320 | 0.735360 | 0.735104 | 0.910144 |

**QKV: stop.** All 18 controls and candidates are exactly equal on the compared
Q/K/V prefixes; full968 x_norm is exact and all physical outputs remain finite.
Mean gain is only 0.001712024 ms / 18 calls, conservative gap 0.001376033 ms,
control drift 0.000671983 ms, candidate drift zero. Within-leg IQR widths are
about 0.0020–0.0030 ms and overlap across controls/candidates. This small static
benefit does not establish room for the extra runtime inactive launch and tail
handling of a deployable candidate. No tile, repeat or production QKV change.

**Out-projection: advance to a bounded deployable candidate.** All 17 controls
are exact against captured outputs. All 17 candidates pass the existing shallow
limits on both 895 and 896 rows, but are not bit-exact. Worst rel_rms is
0.00011054231, minimum cosine 0.999999993894, and maximum absolute difference
4.0 at layer14. All padded rows remain finite. These are local checks, not
official full-model validation.

Mean gain is 0.175999969 ms, conservative gap 0.174783945 ms, control drift
0.002176046 ms, candidate drift 0.000256002 ms. Same-condition separation is
clear. The M896 out-projection workspace is 22,283,008 bytes; QKV uses 131,200.
The comparison changes both implementation (Torch versus cfg0) and M; changed
Stream-K scheduling/workspace may contribute. Do not attribute the speedup only
to omitting one eighth of row tiles, or count it as end-to-end gain.

Both sites completed their single planned ABBA and all 120 samples are retained.
GPU ownership was returned immediately after the successful process exit.
Tracked raw numerical/timing output:
[`local.json`](../../results/rtx5090-pi05/gpt6-prefix-projections-m896/local.json).

Original untouched artifacts:
- `artifacts/rtx5090-pi05/gpt6-prefix-projections-static-m896.json`
- `artifacts/rtx5090-pi05/gpt6-prefix-projections-static-m896.log`

The earlier FFN file `gpt6-prefix-static-m896.json` was not overwritten.
