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

## One completed ABBA: static headroom remains

Executed from source commit `45269d1`, using the unchanged existing cfg0 library.
Actual host/device prefix counts both equal 895, prompt count 127, and
`mask[896]=-3.00405527047391e38`.

All 51 short GEMMs pass existing shallow tolerances on both compared ranges.
They are not bitwise identical: no call is exact. For rows 0:895, maximum
relative RMS is 4.7112142e-5, minimum cosine 0.9999999988902422, and maximum
absolute difference 0.5. For rows 0:896, the corresponding values are
4.7096198e-5, 0.9999999988909931, and 0.5. All 17 full-M968 down references are
bitwise identical to their captured original down outputs.

| Leg | Median milliseconds for 51 GEMMs |
| --- | ---: |
| A1 M968 | 15.1572475433 |
| B1 M896 | 13.6232957840 |
| B2 M896 | 13.6355838776 |
| A2 M968 | 15.4480638504 |

Mean A-minus-B gain is **1.6732158661 ms**. Control drift is 0.2908163071 ms;
candidate drift is 0.0122880936 ms; min(A)-max(B) is 1.5216636658 ms.
A samples visibly drift upward, and every sample was retained. Even the
conservative separation exceeds the observed control drift, so this fixed
static screen supports investigating the runtime implementation costs.
No additional timing or production change followed this screen.

Both independent workspace allocators hold 22,283,008 bytes. The 51 distinct
weights rotate through 3,422,552,064 bytes (3.1875 GiB), and each call retains
its own captured input and output allocation. These graph/cache conditions
differ from the deployed interleaved model. The measured gain is static
GEMM-only headroom, not an end-to-end saving or a causal hardware attribution.

All 51 per-call numerical rows and 60 raw samples are preserved in
`results/pi05-rtx5090/gpt6-run-01/measurements/prefix-static-m896.json`.
The execution log remains
`/home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-prefix-static-m896.log`.
