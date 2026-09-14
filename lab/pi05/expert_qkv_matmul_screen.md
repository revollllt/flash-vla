# Expert QKV GEMM bounded probe

The bounded GPU screen is complete. This probe does not edit a production backend or route.

## Evidence and hypothesis

The 010 action-expert trace attributes 180 calls to each QKV stage:

| Stage | Sum (us) | Mean per call (us) |
|---|---:|---:|
| prepare | 225.798 | 1.254433 |
| BF16 GEMM | 1686.890 | 9.371611 |
| factor/bias/RoPE/scatter | 217.831 | 1.210172 |

Sources: `results/pi05-rtx5090/gpt6-run-01/profiles/010-expert-report.json`,
and `profiles/010-expert/0_shipped_action_expert.json` in the main checkout.
The attribution uses the QKV call site, not the globally shared kernel name.

The current GEMM is
`cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_nn_align8`:
160 CTAs (trace grid 16 x 10), 128 threads, 96 registers/thread, 37,376 B shared
memory. Thus 160-CTA candidates do not solve an absent-CTA problem.
The hypothesis is that a small MMA pipeline / different reuse pattern can improve
this 50 x 1024 x 2560 GEMM. It is not established by the kernel name alone.

Exactly three tiles, with 4 warps and 3 stages:

| BM/BN/BK | CTA | Question |
|---|---:|---|
| 32/32/32 | 160 | Does a short MMA pipeline help with the same logical output partition? |
| 16/64/32 | 160 | Does the change in A/B reuse help? |
| 16/32/32 | 320 | Does more parallel work offset its extra operand requests? |

All three still pad M to 64. The 16-row tiles request B across four row groups,
versus two for 32-row tiles. Any cache reuse of these requests is unmeasured.
No split-K, fused epilogue, or further tile search is included.

Each weight is 5 MiB; all 18 are 90 MiB, below the 96 MiB L2. Unique A/B/output
bytes are 5,601,280 per call, whose ideal time at nominal 1.792 TB/s is 3.126 us.
This is a zero-overhead bandwidth bound, not a latency prediction. The current
GEMM contribution is 1.687 ms per model; every local 1 us reduction corresponds
to 0.18 ms by call-count multiplication, which is not measured E2E savings.
The preceding dual-FFN candidate's local 0.31-0.33 ms extrapolation became about
0.0985 ms in deployment ABBA; its cause is unresolved.

## Protocol

- Record the first real QKV input, prepared BF16 activation, projection, factor,
  RoPE values, and Q/K/V for each distinct layer weight from the selected shipped
  deployment. Record the actual imported deployment plan and revision.
- Keep the deployed prepare and finish kernels, including the intermediate BF16
  projection before FP32 factor, bias, rotation, and final BF16 scatter.
- Verify the control and each tile on all recorded layers using the existing
  shallow rel-RMS/cosine tolerances for projected, Q, K, V, and factor.
  Save per-layer diagnostics and compiled PTX/resources. Stop on any numerical
  or execution failure; exceptions are recorded and re-raised.
- For the cache-pressure screen, use the original 18 weights then 18 cloned
  weights: 180 MiB of distinct storage. A and B use the identical addresses,
  order, inputs, and shared projected output. Cloning happens before timing.
  This is an explicit local cache-pressure experiment, NOT the deployment
  sequence. No copy or cache flush runs in its timed GEMM graph.
- Run one 15-sample ABBA per tile using the existing graph timer. Report raw
  samples, both control/candidate leg medians, their drift, average gain, and
  conservative leg gain. A negative conservative gain indicates overlapping
  legs even when the mean gain is positive.
- If the best tile has positive mean local gain, run one further ABBA of that
  tile versus control on the original 18-weight complete QKV chain. Validate
  the chain before timing. This 90 MiB sequence may be warm and omits all
  interleaved non-QKV deployment work; it is also not E2E evidence.
- If all three lose, preserve the negative result and stop. No automatic
  expansion of configs or production changes.

The script uses `lab/pi05/cutlass_gemm_screen.py::samples_ms`; this helper is
already present in base `eb8c16a` (original helper commit `9b6449e`).

## Run after an exclusive GPU/JIT slot is granted

```bash
cd /home/ubuntu/flash-vla-gpt6-backbone
source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
set -o pipefail
/home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.expert_qkv_matmul_screen \
  --source-checkout /home/ubuntu/flash-vla --seed 42 \
  --option converted_checkpoint=/home/ubuntu/models/pi05_belt_cup_pytorch \
  --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --output /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-expert-qkv-matmul-screen.json \
  2>&1 | tee /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-expert-qkv-matmul-screen.log
```

## Measured result

The fixed screen completed against imported deployment
`cb927cfde6221691b00349f6c1186c1afafa7bc8`, QKV route `fused-qkv`,
Triton 3.7.1. All three tiles were elementwise identical to the control for
projected, Q, K, V, and factor on the first captured call of each of the 18 layers: max absolute
and relative RMS errors were zero. This is not a check of all ten denoising steps.

Each entry below is a leg median in microseconds per call. All 15 raw samples
per leg remain in the JSON; the first sample was retained, not discarded.

| Tile | A1 | B1 | B2 | A2 | Mean A-B | Conservative min(A)-max(B) |
|---|---:|---:|---:|---:|---:|---:|
| 32/32/32 | 9.9884 | 9.1351 | 9.1013 | 10.0018 | 0.8769 | 0.8533 |
| 16/64/32 | 9.9858 | 9.1049 | 9.0800 | 9.9573 | 0.8791 | 0.8524 |
| 16/32/32 | 9.9929 | 8.5680 | 8.5698 | 9.9529 | 1.4040 | 1.3831 |

The first two configurations are similar within their observed leg drift.
The sole winner is 16/32/32. Its A drift is 0.0400 us and B drift 0.0018 us.
It uses 320 CTAs, 40 registers/thread, 0 spills, and 6,144 B shared memory.
The other tiles use 38/40 registers and 8,192/10,240 B shared memory,
respectively, also without spills.

For the original 18-weight complete QKV chain:

| A1 | B1 | B2 | A2 | Mean A-B | Conservative min(A)-max(B) |
|---:|---:|---:|---:|---:|---:|
| 11.6107 | 9.9236 | 10.2329 | 11.1396 | 1.2969 | 0.9067 |

This chain has larger drift: A 0.4711 us, B 0.3093 us. Every B leg remains faster
than every A leg, but the exact magnitude is less stable than the cache-pressure
screen. Neither sequence proves deployed E2E savings or identifies the cause of
the drift. No additional timings or configs were run.

The winner's emitted PTX contains
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (lines 165,168),
then explicit `cvt.rn.bf16.f32` (205-208) before the projected global store
(line 250). The following native finish remains separate.

Raw artifacts in the main checkout:

- `artifacts/rtx5090-pi05/gpt6-expert-qkv-matmul-screen.json`
- `artifacts/rtx5090-pi05/gpt6-expert-qkv-matmul-screen.log`
- `artifacts/rtx5090-pi05/gpt6-expert-qkv-matmul-screen-{32x32x32,16x64x32,16x32x32}.ptx`

The GPU/JIT slot was released immediately after the successful process exit.
