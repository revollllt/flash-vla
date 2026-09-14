# Expert attention: FP32-score QK candidate

The selected 32x32x64 QK passes nine actual attention call sites with bitwise
matching FP32 logits, BF16 probabilities and final output. At the complete
attention boundary, including identical Q reset, the candidate saves about
0.909 us per call locally. Deployed end-to-end latency is not established.

## Bounded hypothesis

The original QK is BF16 Q(400,256) times transposed BF16 K(1018,256), writing
FP32 logits(400,1018). K transpose is a view. Q/out aliasing is handled by
capturing Q before the real invocation and resetting it before every timed
complete-attention invocation. The score layout, runtime mask, native softmax,
materialized BF16 probabilities and torch PV all remain unchanged.

Round-010 attribution had one QK launch per call, with no copy/reduction:
180 calls, 804.084 us total, 4.448 us median.
Kernel name: cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_64x1_tn_align2,
grid (104,4,1), block (128,1,1), 56 registers/thread, 9216 bytes shared memory.
QK has 208486400 conventional FLOPs and 2354816 logical bytes:
Q 204800, K 521216, FP32 logits 1628800. These are not measured DRAM bytes.

Only three fixed tiles were tried, all four warps and three stages:

- 32x32x64: 416 CTAs, four K iterations.
- 32x64x64: only BN changes; 208 CTAs, four K iterations.
- 32x64x128: only BK changes; 208 CTAs, two K iterations.

The hypothesis was that a fixed tile/loading path could improve scheduling
and reuse. No launch can be deleted here. M/N tails are masked directly;
K=256 divides every tested BK. No packing, padding, scratch format change,
softmax fusion or autotune is involved.

## One-pair screen

Actual belt-cup step 0/layer 0, seed 42, source
d68d2efad845ee93fa6345b624e216c326aac3d7.
All three candidates produce bitwise matching FP32 logits on this pair.

Warm-cache graph ABBA used the same immutable Q/K, separate overwritten logits,
64 calls per graph, 5 warm and 30 measured repeats. JIT and preparation precede
timing. No clocks were locked. Times below are microseconds per QK.

| Tile | Torch A1 | Triton B1 | Triton B2 | Torch A2 |
|---|---:|---:|---:|---:|
| 32x32x64 | 4.42800 | 3.54475 | 3.67050 | 4.62000 |
| 32x64x64 | 4.62650 | 3.70275 | 3.70750 | 4.62525 |
| 32x64x128 | 4.62275 | 4.62300 | 4.65250 | 4.62200 |

Both BK64 candidates clearly improve on their controls; BK128 does not.
The first tile's candidate blocks drift by 0.126 us and its controls by
0.192 us. Their separation from the controls is still much larger, but the
small difference between the two BK64 tiles is not claimed as robust.
32x32x64 was selected for the bounded confirmation.

The screen does not support attributing gain to fewer CTAs or a larger N
tile. The selected candidate still uses 416 CTAs. Its generated instructions
are recorded, but no counters establish which scheduling or memory mechanism
causes the observed speed difference.

## Nine-pair complete-attention confirmation

Capture source bf2a9ba86d5e25d444583b234fe3dd3c346ed998,
checkpoint kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha, seed 42.
Selected steps 0/4/9 crossed with layers 0/8/17. Q/K/V/mask are saved before
each original invocation, so Q is never sampled after out=Q overwrites it.
The original model output continues to feed all later calls during capture.

For each of the nine pairs, FP32 logits, BF16 probabilities, and attention
output are bitwise equal to the existing fused-attention backend. This is
evidence for these actual pairs, not a blanket numerical guarantee.

Both timed sides reset Q from the saved input and use out=Q. The candidate
changes only QK; its subsequent native softmax and torch.mm(P,V) are unchanged.
The callable cycles through nine pairs and the graph repeats this sequence
16 times. It uses fresh capture streams, 5 warm and 30 measured repeats.

| ABBA | Median us per 9 calls | IQR us |
|---|---:|---:|
| control A1 | 118.31400 | 118.07150–118.32850 |
| candidate B1 | 110.12600 | 110.11750–110.13850 |
| candidate B2 | 110.13300 | 110.11750–110.15800 |
| control A2 | 118.31500 | 118.30550–118.32600 |

Both candidate IQRs lie below both control IQRs. This gives 8.181–8.189 us
per nine calls, about 0.909–0.910 us per call. The two control medians differ
by 0.001 us. This is a warm working set below L2 capacity, without L2 flush;
it does not reproduce all intervening full-model traffic.

A separate diagnostic process records one Q reset as DtoD memcpy, then the
_qk kernel at grid (13,32,1), block (128,1,1), the original softmax, and the
original PV GEMM plus split-K reduction. Generated QK PTX contains
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32.
Trace durations are not used for the timing conclusion.

## Artifacts and reproduction

The first stage uses lab/sm120/pi05_attention_qk_triton_probe.py; the
confirmation uses lab/sm120/pi05_attention_qk_triton_confirm.py.
Run from /home/ubuntu/flash-vla with artifacts/rtx5090-pi05/gpt6-env.sh
sourced, PYTHONPATH=$PWD/src:$PWD and the main .venv/bin/python.

Both scripts accept prepare --snapshot PATH --checkpoint
/home/ubuntu/models/pi05_belt_cup_pytorch --checkpoint-id
kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha.
Then run time --snapshot PATH --out RESULT.json. Only the confirmation has
trace --snapshot PATH --out dispatch.json.

Snapshots are retained at:
- /home/ubuntu/flash-vla-gpt6-qkv/artifacts/rtx5090-pi05/attention-qk-first-pair.safetensors
- /home/ubuntu/flash-vla-gpt6-qkv/artifacts/rtx5090-pi05/attention-qk-nine-pairs.safetensors

local.json and representative.json retain all metrics and raw timing samples.
dispatch.json retains the dispatch and exact MMA lines, with paths to the
untracked raw trace/PTX. Production routing is outside this lab experiment.

## Optional backend for serial integration

The follow-up source is
src/flash_vla/hardware/nvidia/rtx5090/pi05/backends/triton_qk_attention.py.
It specializes the chosen QK to 32x32x64 and imports the existing Target's
lazy native softmax library. The original fused_attention.py remains the
control. FP32 logits and BF16 probability scratch roles, runtime mask,
softmax, torch PV and output aliasing Q remain unchanged.

This optional production wrapper was prepared under a CPU-only window.
AST and wrapper declaration with CUDA_HOME unset and no visible CUDA device
pass. The final fixed-constant production function has not yet been JIT
compiled or measured. Registry/plan routing, official alignment and deployed
E2E ABBA belong to the serial model loop. No local timing extrapolation has
been counted as a deployed gain.
