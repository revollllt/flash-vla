# Fixed warpN16/count4 expert FFN down screen

This is one joint layout candidate: CTA32x64x32, warp32x16x32, eight stages,
and a candidate-specific four-element rounded epilogue. The control remains
CTA32x64x32/warp32x32x32/count8 in the same native library. Scalar BF16 roundtrip
and separate FP32 multiply/add are retained. No production route or vendor edit.

The [CPU screen](../../results/rtx5090-pi05/gpt6-expert-down-warpn-screen/README.md)
records why warpN alone cannot instantiate, count4 coverage/reduction arithmetic,
the missing two-warp unit throughput measurement, and the performance uncertainty.

The fixed probe reuses the M16 probe's validation and timer logic with only
candidate symbols/layout metadata changed. Apply cutlass_warpn16_down.patch in
an isolated checkout, then use the existing native loader. Run coverage before
actual model loading. Coverage includes three analytical constants over all 50
rows plus one launch/type solely for grid/register/shared metadata. Capture
compiler resources and inspect candidate SASS for the BF16 roundtrip and separate
FMUL/FADD before actual validation.

Actual validation captures all 180 belt-cup calls, retaining 18 weights,
saved inputs/gates/residuals and original outputs. It compares both routes with
existing shallow tolerances, checks same-mainloop decomposition on 0/90/179,
then runs one reset-inclusive ABBA with 15 raw samples/leg. Both routes share
identical tensor addresses and reset every residual before the call. This is a
local sequence; no cold-cache or end-to-end claim. Stop on failure or negative/
unclear timing. This candidate jointly changes warp partitioning and epilogue
vector width, so it cannot isolate the effect of warp count.

Commands from this worktree, after the root grants the serial GPU window:

```sh
source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
export CUTLASS_DIR=/home/ubuntu/flash-vla/third_party/cutlass
/home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.cutlass_warpn16_down \
  --phase coverage \
  --output results/rtx5090-pi05/gpt6-expert-down-warpn-screen/coverage.json
/home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.cutlass_warpn16_down \
  --phase actual --seed 42 \
  --option converted_checkpoint=/home/ubuntu/models/pi05_belt_cup_pytorch \
  --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --output results/rtx5090-pi05/gpt6-expert-down-warpn-screen/actual.json
```
