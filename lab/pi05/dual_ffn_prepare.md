# Fixed RMS-prelude dual FFN experiment

The only candidate folds native AdaRMS into the existing dual-dot suffix.
The dot tile stays 16x64x32, four warps, three stages, unchanged Packed layout,
enable_fp_fusion=False, enable_reflect_ftz=False, and original bias/tanh math.
No production route changes.

A 16x256 FP32 sumsq tensor accumulates four strided loads at col+i*256.
The four-term order matches the native code; the subsequent cross-column
reduction tree may differ and requires actual factor checks.
libdevice.rsqrt maps to __nv_rsqrtf. Its result rounds to BF16 before use.
Inside each original K32 step, raw X times that factor rounds to BF16, then
the product with FP32-expanded BF16 scale rounds to BF16 before both dots.
Each FP32 projection still rounds to BF16 before FP32 bias. Only N tile zero
writes the shared factor output, once for each valid row.

The prelude has approximately 32 FP32 sumsq values plus 32 raw values per
thread, before addressing/temporaries; those arrays need not overlap the dot
accumulators' lifetime. This is an estimate, not a register prediction.
All 64 N tiles repeat RMS and normalization: 3,276,800 valid RMS elements/call
(4,194,304 logical slots with M padding) and an extra 6.25 MiB logical X read.
A's existing direct cp.async path cannot simply transport scaled data.
Weight async pipelining, actual registers, spills, and all BF16 boundaries
must be checked before numerical execution. The 256-CTA grid is unchanged.
The profiled native prepare costs only 1.45176 us/call, which is a gross
opportunity rather than an expected saving.

The compile phase saves baseline/candidate resources, PTX and TTGIR under new
independent names. A compile error, spills, or missing required BF16 boundary
stops the experiment without a changed tile, reduction, flags, or retry.

Actual capture clones all 180 call-site X and scale inputs before the original
call, and Out/Factor after it. It retains the immutable 18 weight pairs and
per-step bias references. A single lab packing of those pairs supplies both
routes. Control includes native prepare plus original dual_dot; candidate
uses the fused kernel. Both write the same Out and Factor addresses and are
checked against the actual captured outputs under existing shallow tolerances;
exactness is also reported for each output. Neither output needs a reset.

Only passing actual checks enter one 15xABBA, with 180 calls per graph in their
original 18-weight order. All 60 samples remain. Gain no greater than drift
stops the route; a positive result is reported before production integration.
The isolated sequence retains the 288 MiB packed-weight rotation but omits
other model operations, and is not a deployed latency measurement.

Use the deployed Python/source path with this isolated script, so control
loads the existing native libraries from the deployment checkout:

```sh
source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
export PYTHONPATH=/home/ubuntu/flash-vla/src:/home/ubuntu/flash-vla
/home/ubuntu/flash-vla/.venv/bin/python /home/ubuntu/flash-vla-gpt6-ffn/lab/pi05/dual_ffn_prepare.py \
  --phase compile \
  --output /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-dual-prepare-compile.json
/home/ubuntu/flash-vla/.venv/bin/python /home/ubuntu/flash-vla-gpt6-ffn/lab/pi05/dual_ffn_prepare.py \
  --phase actual --seed 42 \
  --option converted_checkpoint=/home/ubuntu/models/pi05_belt_cup_pytorch \
  --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --output /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-dual-prepare-local.json
```

## Outcome: reject this fixed candidate

First compilation succeeded. A used 64 registers and 18,432 shared bytes;
B used 72 registers and 19,584 shared bytes. Both had zero spills.
The required factor, normalized, scaled-A and projection BF16 boundaries
remained in compiler output. The two successive rounded products lowered to
separate mul.rn.bf16 instructions. Factor stores were guarded by N tile zero.

The compiler retained two-buffer asynchronous global-to-shared copies for
raw X, scale and both weight branches. It then read raw X from shared memory,
performed the two rounded products, and wrote the resulting A back to shared
memory before ldmatrix. Thus the initial risk of losing all A async loading
did not occur; added preprocessing and shared work still remained.

Against deployment source 7473829, both routes were bitwise equal to all 180
captured actual Out and Factor results: maximum relative RMS and maximum
absolute error were zero. There were 18 packed pairs (288 MiB), packed once
for both lab routes. No numerical failure preceded timing.

| Leg | Median microseconds per complete call |
| --- | ---: |
| A1 native prepare + original dual-dot | 15.8597336875 |
| B1 fused prepare + dual-dot | 16.4250665241 |
| B2 fused prepare + dual-dot | 16.4414220386 |
| A2 native prepare + original dual-dot | 15.8903108703 |

Mean A-minus-B was **-0.5582220025 us/call**. The candidate was slower by
approximately 0.10048 ms over 180 calls, with A/B drift only
0.0305771828/0.0163555145 us/call. Conservative min(A)-max(B) was
-0.5816883511 us/call. All 60 samples, including high first samples, remain.

Reject this fixed candidate and stop. No reduction, tile, stage, flag or
tolerance changes, extra timings, profiler work, or production integration
followed. The comparison does not attribute the slowdown to a specific
hardware bottleneck and is not a deployed latency measurement.

Tracked evidence under results/pi05-rtx5090/gpt6-run-01/measurements:

- dual-prepare-compile.json: both resource records and full PTX/TTGIR paths.
- dual-prepare-resources.txt: selected compiler excerpts with original lines.
- dual-prepare-rejected.json: all 180 per-call numerical checks and 60 samples.

Complete stdout/stderr logs remain under the ignored artifact directory as
gpt6-dual-prepare-compile.log and gpt6-dual-prepare-local.log.
