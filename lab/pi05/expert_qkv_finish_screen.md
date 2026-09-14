# Expert QKV finish fusion: one-tile probe

The fixed GPU probe is complete. No production backend or route changes
are included in this experiment.

## Fixed implementation

Use the previously measured 16x32x32 GEMM tile, 4 warps, 3 stages:
320 CTAs for M50 K1024 N2560. The mainloop still loads the same contiguous
weights and uses one FP32 dot accumulator. After the mainloop:

1. Convert FP32 accumulator to BF16 RN, then back to FP32.
2. Multiply the BF16-derived factor in FP32, then add BF16-derived bias in FP32.
3. Split each adjacent pair via reshape (16,16,2) and tl.split.
4. For Q/K, evaluate even*cos - odd*sin and odd*cos + even*sin separately.
5. Round the final Q/K/V values to BF16 and write the existing output layouts.

Both enable_fp_fusion and enable_reflect_ftz are false. No factor or bias is
moved before the intermediate BF16 rounding. V receives factor and bias, but
no RoPE. The table source is the actual native finish in
`src/flash_vla/hardware/nvidia/rtx5090/pi05/backends/fused_qkv.cu`.

| Projection columns | Output | CTAs |
|---|---|---:|
| 0..2047 | Q[row*2048 + col] | 256 |
| 2048..2303 | K[row*256 + col-2048] | 32 |
| 2304..2559 | V[row*256 + col-2304] | 32 |

All branch boundaries align with the 32-column tile. The caller already passes
K/V suffix pointers, as shown in the Pi0.5 pipeline:
`k=kv_k[layer][prefix_len:]` and `v=kv_v[layer][prefix_len:]`.
The candidate does not add a second prefix offset. RoPE uses the same
`pair % 128` coefficient pair for all nine Q/K heads.

This removes one finish launch and the 256,000-byte projected store/reload.
The removed write+read is 512,000 B/call or 87.89 MiB/180 calls. The 010 finish
duration was 1.210 us/call, but the arithmetic moves into the GEMM: it is not all
recoverable time, and its read cost must not be counted again. More registers,
an extra layout conversion, or a longer epilogue may erase the gain. The current
GEMM's 40-register/6-KiB footprint does not predict this fused kernel's resources.

## Correctness and timing

The selected deployment is imported from --source-checkout; its revision and
actual plan are recorded. Capture the first real call of each of the 18 expert
layers, including the original x, scale, weight, bias, RoPE, expected Q/K/V and
factor, the real prefix K/V values, and their actual suffix pointer offsets.

Construct one K/V cache per layer from those prefix values plus a 50-row suffix.
Both routes use the same weight/input/output addresses and the same unchanged
native prepare. Control uses the main checkout's triton_qkv._qkv_mm followed by
native finish. Candidate uses qkv_mm_finish. The shared scaled activation,
factor and Q buffers are also identical between routes.

Before timing, verify control and candidate on all 18 layers with existing
shallow rel-RMS/cosine tolerances for Q/K/V/factor and exact prefix equality.
Save numerical diagnostics and prefix checks. Any failure is saved and raised,
with no timing of a failing candidate. Save compiled PTX, registers, spills and
shared memory for inspection.

Then run exactly one complete-chain ABBA with 15 raw samples per leg through
the existing graph timer. No cache/input copies occur in timed graphs. Report
all leg medians and both control/candidate drift. The 18 weights total 90 MiB,
below 96 MiB L2: this same-condition local chain omits interleaved non-QKV work
and is not the deployment sequence or E2E evidence. There is no additional
cache-pressure sweep or tile search. A non-positive local result is recorded
and stops the experiment.

Dependencies are already in branch base `0dcb021`:
`lab/pi05/cutlass_gemm_screen.py::samples_ms` and the production control
`src/flash_vla/hardware/nvidia/rtx5090/pi05/backends/triton_qkv.py`.

## Run only after the exclusive GPU/JIT slot is granted

```bash
cd /home/ubuntu/flash-vla-gpt6-backbone
source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
set -o pipefail
/home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.expert_qkv_finish_screen \
  --source-checkout /home/ubuntu/flash-vla --seed 42 \
  --option converted_checkpoint=/home/ubuntu/models/pi05_belt_cup_pytorch \
  --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --output /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-expert-qkv-finish-screen.json \
  2>&1 | tee /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-expert-qkv-finish-screen.log
```

## Measured result

The fixed probe completed with imported deployment
`bf2a9ba86d5e25d444583b234fe3dd3c346ed998`, QKV route `triton-qkv`,
and Triton 3.7.1. Both control and candidate were elementwise identical to
the captured Q/K/V/factor on the first call of all 18 layers: max absolute
and relative RMS errors were zero. All K/V prefix checks passed. The actual
prefix is 968 rows, and both suffix offsets are 247,808 BF16 elements;
the candidate used the already-offset pointer without applying the offset again.

One complete-chain ABBA, with 15 raw samples per leg:

| A1 (us/call) | B1 | B2 | A2 |
|---:|---:|---:|---:|
| 10.807111 | 9.637333 | 9.630222 | 10.775111 |

Mean A-B is 1.157333 us/call; conservative min(A)-max(B) is 1.137777 us.
A leg drift is 0.032000 us and B drift is 0.007111 us. All raw samples, including
each leg's first sample, are retained. The 180-call multiplication is about
0.208 ms locally; this is not measured deployment E2E savings. No extra timing
or configuration was run. The GPU/JIT slot was released after successful exit.

The fused candidate remains at 320 CTAs, 40 registers/thread, zero spills,
and 6,144 B shared memory, matching the separate GEMM's reported footprint.
Its PTX confirms the required ordering:

- Lines 178/181: BF16-input FP32-accumulator mma.sync.m16n8k16.
- Lines 228-231: explicit FP32-to-BF16 RN conversion.
- Lines 256-260: BF16 back to FP32, followed by factor multiplication
  (274-275) then bias addition (282-283), as separate RN f32x2 operations.
- Lines 378-394: separate RN products and subtract/add for adjacent RoPE pairs.
- Lines 419-420 and 477-479: final BF16 RN conversions before Q/K/V stores.
- No fma instruction appears in the emitted PTX.

The newer validated 019 expert profile independently attributes 223.818 us to
prepare, 1476.047 us to GEMM, and 217.604 us to finish across 180 calls.
The finish time is almost identical to 010; neither profile alone proves the
fused candidate's E2E gain.

Raw artifacts in the main checkout:

- `artifacts/rtx5090-pi05/gpt6-expert-qkv-finish-screen.json`
- `artifacts/rtx5090-pi05/gpt6-expert-qkv-finish-screen.log`
- `artifacts/rtx5090-pi05/gpt6-expert-qkv-finish-screen.ptx`
