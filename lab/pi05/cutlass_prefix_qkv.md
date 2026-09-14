# Fixed existing cfg0 reuse for prefix QKV

The exact test is BF16 M968/K2048/N2560, alpha=1, beta=0, over all 18 real
backbone layers. The existing fused-prefix-qkv RMSNorm and adjacent-pair
RoPE/scatter surround a materialized BF16 projection. Only torch.mm is
replaced by cutlass_backbone._Plan and its current same-library cfg0.
No CUDA, vendor, native ABI, production factory or route changes are needed.

The 010/021 traces attribute 887.801/896.322 us to the 18 QKV GEMMs.
The original cuBLAS symbol is
cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_nn_align8;
torch.mm does not request a ReLU despite this generic symbol's name.
Its grid is 32x5x1, block 128, 242 registers/thread and 81920 shared bytes.
Existing cfg0 has CTA128x128x64, warp64x64x64, three stages, 254 registers
and 98304 shared bytes in the retained compiled implementation.
Both have 160 logical output tiles here; this does not predict performance.

The earlier negative cfg0 out-projection test was M968/K2048/N2048 with
beta=1. Original twelve-config screening tested N16384 gate/up and
K16384/N2048 down. The old Pi0 QKV benchmark used M768 and a different
hand-written kernel. No retained test of this exact Pi0.5 QKV cfg0 shape
was found before preparing this bounded experiment.

## Fixed experiment

The script records x before each actual shipped prefix call; all later
layers still consume the original route's result. It retains each layer's
weight, RoPE, and captured Q/K/V/x_norm outputs. There are 18 distinct
10 MiB weights (180 MiB total, above the 96 MiB L2).
The retained norm output is its valid first M rows.

Control is the original fused-prefix-qkv wrapper. Its scratch callback
returns the same BF16 projected allocation used by the candidate.
Both routes also share working normed/Q/K/V addresses and each immutable
saved input address. Candidate plans own their retained tensors and use
the existing runner Scratch workspace; all 18 plans are initialized before
timing, and scratch is frozen after numerical checks. There is only one
cfg0 shared library in the process.

For all 18 layers, compare both routes' Q/K/V/x_norm against actual captured
outputs. Exact equality is reported directly. Any non-exact output uses
the unchanged six error_metrics and existing shallow tolerances.
Each layer is saved before a numerical failure raises; no timing follows
a failure. Exact outputs avoid unnecessary quantile calculations.

After numerical success, one A1/B1/B2/A2 has 15 raw samples per leg.
Each graph contains the complete chain for all 18 distinct layer inputs.
All outputs are overwritten; no reset is needed. This is an isolated
same-address weight rotation without an L2 flush, not full-model traffic.
All first measured samples are retained. A negative result or a gain that
does not exceed drift stops this fixed reuse candidate without another tile.

Run from the experimental checkout, after sourcing
/home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh and setting
CUTLASS_DIR=/home/ubuntu/flash-vla/third_party/cutlass:

~~~sh
/home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.cutlass_prefix_qkv --seed 42 \
  --option converted_checkpoint=/home/ubuntu/models/pi05_belt_cup_pytorch \
  --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --output /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-prefix-cfg0-screen.json
~~~

The script imports the existing lab/pi05/cutlass_gemm_screen.py timing helper,
which is present in the base revision. Initial checks are py_compile and a
scoped source diff check. The following actual numerical and timing evidence
was obtained inside the explicitly granted GPU window.

## Measured result: stop

Source and recorded engine revision:
91754966653991f2fc2c9b85301d8915bb73ef8c, based on 86f58f3.
The captured shipped plan is retained 021, including fused-prefix-qkv and
the original cutlass-vision FFN up. Seed 42 and converted belt-cup checkpoint
identity/options are saved in the raw result.

All 18 control and all 18 candidate Q, K, V and x_norm tensors are exactly
equal to the corresponding captured outputs. No non-exact fallback metric
calculation was needed. The candidate used only
/home/ubuntu/flash-vla-gpt6-backbone/.cache/cuda_ext/rtx5090_pi05_cutlass_backbone/libcutlass_backbone.so.
Its reusable workspace is 131200 bytes and the common BF16 projected buffer
is 4956160 bytes. The 18 distinct weights total 188743680 bytes (180 MiB).
The run log contains no CUTLASS ptxas rebuild output; the existing cfg0 cache
was reused.

| Leg | Total median ms / 18 calls | Total IQR ms |
|---|---:|---:|
| A1 current | 1.043584 | 1.043168–1.044944 |
| B1 cfg0 | 1.046688 | 1.046224–1.047616 |
| B2 cfg0 | 1.047520 | 1.045936–1.048368 |
| A2 current | 1.052352 | 1.051472–1.052592 |

Mean-of-leg-medians apparent gain is only 0.000863969 ms, compared with
0.008767962 ms control drift and 0.000832081 ms candidate drift.
The conservative separation min(A)-max(B) is negative, -0.003936052 ms.
Both candidate medians are slower than A1. The mean gain therefore does not
establish an improvement above observed drift; no mechanism is assigned.

This fixed cfg0 reuse stops without production integration, another tile,
another comparison or a deployed end-to-end run. All 60 timing samples
retain their first measured value; no observations were discarded.

Tracked raw result:
results/rtx5090-pi05/gpt6-prefix-cfg0/local.json.
Original stdout and JSON:
- /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-prefix-cfg0-screen.log
- /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-prefix-cfg0-screen.json

GPU/JIT/NVCC ownership was returned immediately after successful process exit.
