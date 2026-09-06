# Unit: coop — cooperative launch, the grid barrier, and what it replaces

**Constants** `coop.lat.dev.sync`, `coop.ctas.dev.max`, `coop.ratio.dev.relaunch` ·
**Probe** `probes/units/coop_launch/coop_launch.{cu,py}`

`launch.lat.dev.ramp` says every kernel starts ~1.24 µs in debt. The decision to
reject cooperative launch for this repo's decoder rested on the assumption that
a grid barrier costs *about the same* — an assumption that was load-bearing and
never measured. It is now, and it holds.

```
sbatch sbatch/pi05_cuda.sh .claude/skills/hardware-unit-test/probes/units/coop_launch/coop_launch.py \
    --json profiles/hardware-unit-test/coop.json
```

Every name is the vendor's: `cooperative_groups::grid_group`, `this_grid()`,
`grid_group::sync()`, `cudaLaunchCooperativeKernel`,
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaDevAttrCooperativeLaunch`.

## The three numbers

**A grid barrier costs 1.09 µs and barely scales** (`coop.lat.dev.sync`).
Measured as the difference between two modes of ONE kernel on ONE grid, both
launched cooperatively, so the barrier is the only thing that moved:

| blocks | threads | with sync | without | ns per grid_sync | cycles |
|---|---|---:|---:|---:|---:|
| 32 | 256 | 2187.0 µs | 55.6 µs | **1065.7** | 2110 |
| 66 | 256 | 2206.6 µs | 55.6 µs | **1075.5** | 2129 |
| 132 | 256 | 2233.8 µs | 55.8 µs | **1089.0** | 2157 |

2000 barriers per launch, median of 7. A 4× change in block count moves the cost
1.02×, so this is a fixed latency, not a per-block cost.

**The co-resident limit is exactly the occupancy query** (`coop.ctas.dev.max`).
`cudaLaunchCooperativeKernel` refuses rather than deadlocks, so the bound is
enumerated from the API's own answer:

| blocks | accepted | rc |
|---|---|---|
| 1055 | yes | 0 |
| **1056** | **yes** | 0 |
| **1057** | **no** | 720 `cudaErrorCooperativeLaunchTooLarge` |
| 2112 | no | 720 |

`max_active_blocks_per_sm × SMs` = 4 × 132 = 1056 at 256 threads, and the
refusal lands on it exactly. This is the opposite of `cluster.count.max`, where
placement did *not* follow the SM count and had to be found behaviourally.

**A relaunch is only 1.29× a grid_sync** (`coop.ratio.dev.relaunch`). Same grid,
same loop body, once as barriers inside one cooperative launch and once as 64
ordinary launches replayed from a CUDA graph:

| | ns |
|---|---:|
| one `grid_sync`, 132 blocks | 1089 |
| one device-side relaunch | 1400 |

Cross-check: `launch.lat.dev.ramp` measured a launch at 1240 ns by a different
route and a different harness; this reads 1400 with a 31-iteration body
included. The two agree within that body's cost, which is what says the
comparison is between two real things.

## What this settles

**The assumption held.** A grid barrier is comparable to a launch, not an order
of magnitude cheaper, so replacing N launches with N grid barriers saves ~22% of
the launch term and nothing else. A persistent kernel has to earn its keep on
state kept in registers and shared memory across phases — which this unit does
not measure and which no constant here supports yet.

## Isolation and regime

Both arms of the barrier pair are cooperative launches on the same grid with the
same body; only the barrier differs. Nothing here touches memory, so there is no
cold/warm regime to declare. Clocks are unpinnable (rule 10), which is why the
table reports cycles beside nanoseconds.

## Falsifiers

- If the barrier scaled with block count, 32 and 132 blocks would not agree to
  1.02× over a 4× range.
- If the co-resident bound were not the occupancy query, 1056 and 1057 would not
  straddle it exactly.
- If the relaunch arm were measuring host dispatch rather than the device, it
  would not land within a loop body's cost of `launch.lat.dev.ramp`'s
  independently measured 1240 ns. **It did not, at first:** dispatched from
  Python through ctypes the same measurement read 6345 ns per launch, a 5×
  host-side inflation that would have reported the barrier as 5.82× cheaper than
  a launch. The CUDA-graph capture is what makes the number a device number.

## Open gaps

- **One thread block size.** Everything is 256 threads. `coop.ctas.dev.max`
  moves with occupancy by construction, so the *limit* generalises through the
  query, but the barrier's 1.09 µs has not been measured at other block sizes.
- **Zero-skew only.** Every block reaches the barrier at the same time. What a
  barrier costs when blocks arrive skewed — the case a real persistent kernel
  has — is unmeasured, and `cluster.lat.sync` found placement dominates exactly
  that for cluster barriers.
- **Run-to-run spread.** Within one job the per-launch spread reached 12.8% at
  66 and 132 blocks against a 6% floor, while the median reproduced to ~1%
  across three separate jobs. Quote the median; do not read a single launch.
- **No probe for what a persistent kernel actually buys.** This unit prices the
  barrier. Whether keeping state across phases pays for it is the question a
  megakernel decision actually turns on, and nothing here answers it.
