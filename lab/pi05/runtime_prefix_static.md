# Fixed static M896 cfg0 upper-bound screen

This lab probe tests whether the 896-row bucket has enough GEMM-only headroom
to justify a runtime selector implementation. It does not add one.

Capture the 17 actual seed-42 backbone FFNs in original gate/up/down order.
Clone each prepared normed input after the existing FFN call; retain gate/up
weights. Clone down input and residual before its existing call, and retain
its output for an additional control check. Replay each captured input once
through ordinary M968 cfg0 to create the GEMM reference; gate/up references
are replay results, not snapshots of original intermediate GEMM outputs.

Only M changes to 896. Both routes use the existing same-library cfg0 _Plan,
with views sharing each call's A/B input and output address. Two independent
Scratch allocators keep route workspaces disjoint. Outputs are distinct across
the 51 isolated calls, so each down residual can be restored before the graph.
Both routes perform the same full-968-row resets outside the timed graph.
This is an isolated real-input replay, not a composed FFN or model trajectory.

Check that the actual device mask and host count both have 895 valid prefix
rows. Compare all 51 short outputs with M968 reference on both the first 895
valid rows and first 896 bucket rows under existing shallow tolerances. Also
compare each M968 down reference to the captured original down output.
Any failed numerical comparison stops before timing.

Run exactly one ABBA with 15 measured graph replays per leg, keeping all 60
samples. Each graph has 51 GEMMs and uses the existing graph timing helper.
Report total graph medians, mean A-minus-B gain, both within-route drifts and
minimum separation. A nonpositive gain or gain no larger than drift stops the
route. A positive result is only static GEMM headroom: dynamic selection,
extra inactive launches, pointwise work, full-model cache traffic, and changing
mask correctness are absent. Report it before any production implementation.

Run from this worktree with the existing environment:

```sh
source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
export PYTHONPATH="$PWD/src:$PWD"
/home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.runtime_prefix_static \
  --seed 42 \
  --option converted_checkpoint=/home/ubuntu/models/pi05_belt_cup_pytorch \
  --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --output /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-prefix-static-m896.json
```
