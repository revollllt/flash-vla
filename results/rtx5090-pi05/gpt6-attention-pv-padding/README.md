# Expert attention PV padding: rejected dispatch screen

K=1024 padding removes the split-K reduction but selects a slower GEMM.
Padding and copying were excluded from timing, so a complete-chain
implementation has no supported gain. No production code changed.
This result does not rule out other PV GEMM implementations.

## Existing attention attribution

Sources: main Target fused_attention.py / .cu and the retained graph trace
/home/ubuntu/flash-vla/results/pi05-rtx5090/gpt6-run-01/profiles/010-expert/0_shipped_action_expert.json.

Q is BF16 (400,256), K/V are BF16 (1018,256), and output aliases Q.
QK produces FP32 logits. A 256-thread CTA per query applies FP32 scale,
runtime additive mask, max/exp/sum and materializes BF16 probabilities.
PV is torch.mm. K transpose is a view. Prefix/suffix lengths are 968/50.

Every distinctive softmax launch and its adjacent QK/PV/reduction kernels
give the same four-kernel pattern across all 180 attention instances:

| Component | Launches | Total us | Median us |
|---|---:|---:|---:|
| QK | 180 | 804.084 | 4.4480 |
| softmax | 180 | 384.005 | 2.1435 |
| PV GEMM | 180 | 734.742 | 4.0640 |
| PV split-K reduction | 180 | 273.991 | 1.5040 |
| all kernels | 720 | 2196.822 | |

PV including reduction is 45.9%, QK 36.6%, softmax 17.5%.
Summed four-kernel envelopes are 2243.734 us; internal GPU graph gaps total
46.915 us (15.847 / 15.486 / 15.582 us between successive kernels).
These are not measured CPU overhead. There are no copy/memset kernels
within attention. Removing only the reduction and preceding gap offers
at most 0.290 ms per forward before replacement work.

QK uses cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_64x1_tn_align2
at grid (104,4,1). PV uses
cutlass_80_tensorop_bf16_s16816gemm_bf16_128x64_64x3_nn_align2
at grid (14,1,12), then cublasLt::splitKreduce_kernel at grid (8,25,1).

Logical bytes per call: Q/output 204800 each, K/V 521216 each, FP32 logits
1628800, BF16 P 814400. Combined K/V for all 18 layers is 18763776 bytes;
prefix is 17842176 bytes and reused across ten denoising steps.
Intervening expert weights exceed L2, so that size does not prove residency.
P is freshly written before PV; V cache residency is unmeasured.
No measured DRAM-bandwidth claim is inferred from logical sizes.

## Actual-pair experiment

Main source c7e3b125d6184987c38270950cabfc1999d78e73, shipped plan,
checkpoint kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha, seed 42.
One actual step-0/layer-0 materialized P/V pair was captured.
The complete engine identity is retained in local.json.

P padding adds six zero columns; V adds six zero rows. All preparation is
outside timing and profiling. Inputs stay immutable; both cases overwrite
separate outputs. This is a warm-cache repeated-pair screen, not 180-call
cache behavior or end-to-end latency. CUDA graph ABBA uses fresh capture
streams, 64 calls/graph, 5 warm repeats and 30 measured repeats per case.
Clocks were not locked. Diagnostic tracing ran in a separate process.

| ABBA case | Median us | IQR us |
|---|---:|---:|
| K=1018 A1 | 5.58325 | 5.58037–5.59550 |
| K=1024 B1 | 6.92775 | 6.92337–6.93913 |
| K=1024 B2 | 6.92625 | 6.92400–6.93800 |
| K=1018 A2 | 5.77350 | 5.77150–5.78350 |

Both padded IQRs exceed both original IQRs. Original controls drift 0.19025 us;
the nearest comparison remains 1.15275 us slower with padding.
The diagnostic dispatch summary confirms unchanged K=1018 dispatch, while
K=1024 selects one
cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_nn_align8
launch, grid (104,1,1). It eliminates split-K reduction but is slower in the
unprofiled graph measurement. Trace durations do not determine the conclusion.

Results are not bitwise equal: rel_rms 0.00268180, cosine 0.999996404,
max_abs 0.0625, p99_abs 0.0078125. They pass existing shallow limits for this
pair; other layers/steps are untested. The selected GEMM and rounding order
change even though the padded mathematical sum is the same.

A full candidate would additionally copy/zero-fill V in the softmax launch,
adding about 1.04 MB of V copy traffic per call. The negative isolated result
does not justify implementing that production path.

## Reproduction

Run from main so native cache and Python ABI match. The lab script can stay
in the worker checkout; explicit PYTHONPATH selects main source.

    cd /home/ubuntu/flash-vla
    source artifacts/rtx5090-pi05/gpt6-env.sh
    export PYTHONPATH=$PWD/src:$PWD

Run lab/sm120/pi05_attention_pv_padding_probe.py with the main .venv/bin/python.
The script modes are separate processes:

- prepare --snapshot PATH --checkpoint /home/ubuntu/models/pi05_belt_cup_pytorch --checkpoint-id kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha
- time --snapshot PATH --out local.json
- trace --snapshot PATH --out dispatch.json

The actual snapshot is at
/home/ubuntu/flash-vla-gpt6-qkv/artifacts/rtx5090-pi05/attention-pv-padding.safetensors.
The raw diagnostic trace stays at the path in dispatch.json.
Only its small dispatch summary and timing samples are committed.
