# Fixed expert FFN down split factor 8

This candidate retains the deployed cfg9 kernel: CTA32x64x32, warp32x32x32,
8 stages, BF16 inputs and the fully reduced FP32 accumulator followed by
BF16 roundtrip, independent FP32 gate multiply/residual add and BF16 output.
Only the CUTLASS plan argument batch_count changes from heuristic mode1 to
explicit split factor8 for M50/K4096/N1024. Control and candidate use the
same DownGemm type in one shared library; out-projection is unchanged.

The current 021 path has 32 logical output tiles and 128 K32 iterations/tile.
The vendored ThreadblockSwizzleStreamK heuristic selects 160 SK blocks,
one padded 170-block wave and 128 separate reduction blocks (grid298).
This is source arithmetic consistent with prior actual cfg9 grid metadata.
Its 64-thread CTA offers two warps; no measured two-warp MMA throughput
constant or actual per-SM SK-placement distribution is available.

Explicit split8 assigns 256 compute blocks, each with 16 K32 iterations.
With the existing two-CTA shared-memory occupancy and 170 SMs, the source
mapping predicts two padded waves (grid340), no separate reduction blocks,
and deterministic turnstile aggregation before the finishing peer's existing
epilogue. This tests a different concurrency/reduction schedule without
changing the failed warpN16/count4 layout or the rejected M16 tile.
The eight-stage prologue is a larger fraction of the shorter K loop, and
seven contributing peers can increase ordered atomic/fixup cost. More
blocks and no separate reduction wave do not predict a latency gain.
The vendor source and production kernel/route remain unchanged.

The minimal native patch adds only two plan/workspace entry points;
run/destroy use the existing functions. The probe reuses the recorded M16
screen protocol with this one scheduling change. First, three constant
cases must cover every output row exactly. Then record real grid/resources,
capture all180 actual calls and compare both routes against the deployed
outputs under existing shallow tolerances. Compare the candidate rounding
decomposition at calls0/90/179 exactly. After success, run one reset-inclusive
15-sample ABBA over the same inputs/output address and 18 real weights
(144 MiB). Save every first sample. Failure, slowdown or gain within drift
stops without another split factor, tile, E2E or production integration.
This isolated replay is not the deployment cache sequence.

Prepared from main6982545. Apply cutlass_split8_down.patch before running.
Use the main environment/venv with PYTHONPATH pointing to this checkout and
CUTLASS_DIR=/home/ubuntu/flash-vla/third_party/cutlass. Run coverage before
actual; the actual phase takes the same belt-cup checkpoint options and
seed42 as earlier screens. Full commands/raw output will be recorded
after the explicitly serialized GPU window.

## Measured result: reject

Lab source1be5057 plus the recorded native patch was used on the same RTX5090,
CUDA13.1 and main CUTLASS checkout. Compilation and execution had exclusive GPU
ownership. All three constant cases cover every one of the50 rows exactly.
Actual metadata agrees with source scheduling: control grid298, candidate340,
both block64,120 registers/thread,49152 shared bytes, two-CTA/SM shared-memory
occupancy limit. The profiler's grid-derived warps/SM field is not a measured
distribution of active SK blocks across SMs. Workspace grows from2,622,080 to
4,195,328 bytes. Both paths invoke the identical compiled DownEpilogue kernel;
the retained resource excerpt reports zero spills.

All180 real calls pass existing shallow tolerance. Control is exact for180;
candidate is exact for35, max_abs0.25, worst rel_rms0.0000802042443,
minimum cosine0.999999996784. The selected0/90/179 candidate projection plus
separate rounded residual decomposition is exactly equal to its fused result.

Reset-inclusive ABBA median us/call:
A1 10.570666525,
B1 15.137422085,
B2 15.132978227,
A2 10.580088695.
Mean candidate slowdown is 4.559822546 us/call against control
drift 0.009422170 and candidate drift 0.004443857.
All60 samples including every first sample are retained. More compute blocks
and the cooperative reduction path did not improve this fixed local boundary;
no counter-based cause or deployment slowdown is inferred.

This candidate stops without another split factor, kernel layout or E2E run.
Native entry points and the experiment library were removed after collecting
evidence; the exact patch remains reproducible. No production plan changed.

Commands (after sourcing the main environment, with this worktree as PWD and
CUTLASS_DIR pointing to main third_party/cutlass):
- /home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.cutlass_split8_down --phase coverage --output /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-split8-down/coverage.json
- /home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.cutlass_split8_down --phase actual --seed 42 --option converted_checkpoint=/home/ubuntu/models/pi05_belt_cup_pytorch --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha --output /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-split8-down/local.json

Full stdout/stderr stays
in /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-split8-down/coverage.log and actual.log; resource-only trace is coverage.trace.json.
Tracked result files are results/rtx5090-pi05/gpt6-expert-down-split8/coverage.json, local.json and resources.txt.
