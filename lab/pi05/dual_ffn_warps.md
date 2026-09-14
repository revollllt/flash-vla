# Rejected dual FFN eight-warp screen

Only `num_warps` changed from 4 to 8 in the existing `dual_ffn._dual_dot`.
The tile stayed 16x64x32, grid 4x64, stages 3, and both
`enable_fp_fusion=False` and `enable_reflect_ftz=False` were retained.
No production source, route, native library, or precision setting changed.
The source base was `698254577d37b74abf40d09da4baa2379d595b46`.

The prior fixed-tile screen used 4 warps for all three tiles. This experiment
tested whether spreading the same two accumulators and serial activation work
over twice as many warps helped the accepted tile. The 256-CTA grid and total
matrix work were unchanged. Resource counts alone did not establish a bottleneck.

## Compiler evidence

| Variant | Registers/thread | Spills | Shared bytes | Threads/CTA | MMA warp layout |
| --- | ---: | ---: | ---: | ---: | --- |
| A: 4 warps | 64 | 0 | 18432 | 128 | 1x4 |
| B: 8 warps | 40 | 0 | 18432 | 256 | 1x8 |

Static MMA instructions per loop per warp decreased from 8 to 4; total warp-MMA
work stayed the same. Static tanhf bodies per thread decreased from 8 to 4.
The 8-warp PTX retained RN BF16 conversion of both accumulators before FP32
bias additions, and RN BF16 conversion after GELU/product. Its
`add.rn.f32.bf16` expands an already-rounded BF16 operand while adding FP32 bias.
The unchanged libdevice tanhf implementation uses internal FMA instructions.

## Real-input check and timing

The probe captured the first prepared input for each of 18 distinct packed
weight pairs during the actual seed-42 belt-cup action-expert invocation.
It cloned prepared activations before the original call and its output after
the call; original execution and returned compiled-kernel objects were preserved.
Weight and bias references stayed alive. Both A and B were compared with those
captured outputs: all 18 layers were bitwise equal for both variants.

Both variants used the same output address and the same 18-layer weight order.
The output is fully overwritten, so no residual reset applies. Preparation was
excluded from both. Each timing graph contained one call per layer, with four
warmup passes before capture. One fixed ABBA sequence used 15 samples per leg;
all 60 samples, including high first samples, are retained.

| Leg | Median microseconds/call |
| --- | ---: |
| A1 | 14.2311106 |
| B1 | 14.4000004 |
| B2 | 14.4035551 |
| A2 | 14.2080006 |

Mean A-minus-B was **-0.1822222 us/call**: B was about 1.28% slower.
A drift was 0.0231100 us and B drift was 0.0035548 us.
The conservative min(A)-max(B) gain was -0.1955546 us. Reject this fixed
eight-warp candidate and stop; production keeps four warps.

The packed-weight rotation was 288 MiB, larger than the 96 MiB L2.
This isolated suffix sequence omits model interleaving. It does not measure
deployed latency or establish which hardware bottleneck caused the slowdown.
No extra tile, stage, profiling, repeat timing, or deployment test followed.

## Reproduction and evidence

From this checkout, with the existing environment and checkpoint:

```sh
source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
export PYTHONPATH="$PWD/src:$PWD"
/home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.dual_ffn_warps \
  --phase compile --output /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-dual-warps-compile.json
/home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.dual_ffn_warps \
  --phase actual --seed 42 \
  --option converted_checkpoint=/home/ubuntu/models/pi05_belt_cup_pytorch \
  --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --output /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-dual-warps-local.json
```

Tracked evidence under `results/pi05-rtx5090/gpt6-run-01/measurements/`:

- `dual-ffn-w8-compile.json`: compiler resources and full PTX/TTGIR artifact paths.
- `dual-ffn-w8-resources.txt`: compact mapping and rounding excerpts.
- `dual-ffn-w8-rejected.json`: actual plan, all numerical checks, and 60 raw samples.
