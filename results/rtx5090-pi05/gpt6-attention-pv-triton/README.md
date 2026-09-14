# Expert attention: fixed-tile Triton PV screen

The 16x32x64 candidate is faster at the isolated PV boundary and passes the
existing numerical limits on nine actual step/layer pairs. It is ready for
complete-attention testing; production source and routing remain unchanged.

## Hypothesis and controlled tile choices

Current BF16 P(400,1018) times BF16 V(1018,256) invokes a split-K GEMM and
separate reduction. The round-010 reduction takes 1.504 us median.
A single CTA computes its output tile across the complete K dimension,
masks the K tail directly, accumulates in FP32, and stores BF16 with rtne.
It needs neither padded inputs nor scratch. QK and the materialized BF16
softmax probabilities stay outside this experiment.

Source references consulted before implementing: local Triton upstream
python/tutorials/03-matrix-multiplication.py, installed Triton 3.7.1 dot/load/
store definitions, and this Target's existing dual_ffn.py. The small lab
kernel is in lab/sm120/pi05_attention_pv_triton_probe.py.

Only three configurations were tested, all four warps and three stages:

- 16x32x64: 200 CTAs, sixteen K iterations.
- 16x32x128: only BK changes; 200 CTAs, eight K iterations.
- 16x16x64: only BN changes from the first case; 400 CTAs.

CTA counts alone do not establish occupancy or expected speed.
No autotune or additional tile sweep was run.

## Initial same-pair screen

Checkpoint kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha,
seed 42, actual step 0/layer 0 P/V snapshot previously captured from main
c7e3b125d6184987c38270950cabfc1999d78e73. Both implementations read identical
immutable inputs and overwrite separate BF16 outputs.
Warm-cache CUDA graph ABBA used fresh capture streams, 64 calls per graph,
5 warm repeats and 30 measured repeats. No clocks were locked.
JIT, allocation and input preparation precede timing.

| Tile | Torch A1 us | Candidate B1 us | Candidate B2 us | Torch A2 us |
|---|---:|---:|---:|---:|
| 16x32x64 | 5.58400 | 5.00375 | 5.00300 | 5.77250 |
| 16x32x128 | 5.77750 | 5.01450 | 5.02600 | 5.79350 |
| 16x16x64 | 5.81975 | 7.53325 | 7.51900 | 5.79175 |

Both 16x32 candidates beat their corresponding torch controls. Their timing
distributions overlap; these data do not establish a meaningful BK64 versus
BK128 difference. BK64 was selected for the limited confirmation.
The 16x16 output tile was slower and was dropped.

All three results are non-bitwise relative to torch.mm: rel_rms 0.00268180,
cosine 0.999996404, max_abs 0.0625. Existing shallow limits pass.

## Nine actual pairs

The confirmation script captured step 0/4/9 crossed with layer 0/8/17 from
current main cb927cfde6221691b00349f6c1186c1afafa7bc8. It reads the deployed
materialized P and V immediately after each selected attention call.
It does not alter the actual attention output feeding later layers/steps.
This capture revision differs from the initial screen; each comparison
still uses the same saved P/V on its two sides.

All nine pairs pass existing shallow requirements. Worst rel_rms is
0.00274679, minimum cosine 0.999996229, maximum max_abs 0.0625.
None are bitwise equal. No claim is made about accumulated model error.

The ABBA callable cycles through nine distinct pairs. The working set remains
warm and is not a simulation of full-model cache behavior. Logical P/V plus
both output sets total 15706944 bytes, below RTX 5090 L2 capacity. The graph
contains sixteen repetitions of this nine-call sequence; each reported sample
is the time per nine calls. Warm/measurement repeat counts remain 5/30.

| ABBA | Median us per 9 calls | IQR us |
|---|---:|---:|
| torch A1 | 50.786 | 50.7655–50.8125 |
| Triton B1 | 43.946 | 43.8880–43.9950 |
| Triton B2 | 43.915 | 43.8655–43.9780 |
| torch A2 | 51.024 | 50.9875–51.0430 |

Both candidate IQRs lie below both controls. Improvement is 6.840–7.109 us
per nine calls, about 0.76–0.79 us per call. Control drift is 0.238 us per set.
Each PV has 208486400 conventional GEMM FLOPs; the first two medians correspond
to about 36.95 TFLOP/s for torch and 42.70 TFLOP/s for the candidate.
These are local timings, not deployed model throughput.

A separate diagnostic process verifies exactly one _pv launch, grid (25,8,1),
block (128,1,1). Generated PTX contains
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32.
There is no split-K reduction launch. The small dispatch summary includes
the exact MMA lines and paths to raw trace/PTX. It makes no tcgen05 claim.

The next decision requires timing the complete unchanged QK + softmax +
candidate PV chain with actual Q alias/reset behavior. Only if that boundary
wins should the model loop evaluate accumulated correctness and end-to-end
latency. No production integration occurred in this task.

## Reproduction and retained artifacts

Run from /home/ubuntu/flash-vla after sourcing
artifacts/rtx5090-pi05/gpt6-env.sh and setting PYTHONPATH=$PWD/src:$PWD.
Use its .venv/bin/python. Both lab scripts are in the worker checkout
/home/ubuntu/flash-vla-gpt6-qkv/lab/sm120/.

Initial script arguments:

    pi05_attention_pv_triton_probe.py --snapshot /home/ubuntu/flash-vla-gpt6-qkv/artifacts/rtx5090-pi05/attention-pv-padding.safetensors --out /home/ubuntu/flash-vla-gpt6-qkv/results/rtx5090-pi05/gpt6-attention-pv-triton/local.json

Confirmation uses pi05_attention_pv_triton_confirm.py with three separate modes:

- prepare --snapshot PATH --checkpoint /home/ubuntu/models/pi05_belt_cup_pytorch --checkpoint-id kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha
- time --snapshot PATH --out representative.json
- trace --snapshot PATH --out dispatch.json

The actual nine-pair snapshot is
/home/ubuntu/flash-vla-gpt6-qkv/artifacts/rtx5090-pi05/attention-pv-nine-pairs.safetensors.
local.json and representative.json retain raw timing samples and all numerical
metrics. dispatch.json retains the one-launch/PTX evidence. Large raw traces,
PTX and tensor snapshots remain at their recorded paths, outside the commit.
