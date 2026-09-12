# unit: coop — what a device-wide barrier costs

Measured with the `hardware-unit-test` skill's own `coop_launch` probe, run
through `lab/sm120/run_skill_probe.py` so the arch flag becomes `sm_120f`
without editing the skill. The probe contains no sm90-only construct; only its
default target had to change.

```bash
CUDA_HOME=... CUTLASS_DIR=... python3 lab/sm120/run_skill_probe.py coop_launch
```

## Claims

**`[coop.lat.dev.sync]` — one `grid_sync` costs 599 ns.**

| blocks | threads | sync µs | empty µs | ns/sync | cyc/sync | spread |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 256 | 1221.5 | 54.8 | 583.4 | 1705 | 9.7% |
| 66 | 256 | 1221.9 | 54.7 | 583.6 | 1704 | 8.3% |
| 132 | 256 | 1252.6 | 54.9 | 598.8 | 1753 | 9.0% |

Measured as the *difference* between two modes of one kernel on one grid, so
nothing but the barrier moved. It barely changes with grid size, as on sm90.

Against sm90's 1089 ns this is **1.8× cheaper**. The spread, 8–10%, is above
this machine's 6% noise floor, so treat the third digit as noise.

**`[coop.ctas.dev.max]` — 1020 blocks, and the launch refuses 1021.**

`cudaLaunchCooperativeKernel` accepted every grid up to 1020 at 256 threads and
refused 1021 and above with `cudaErrorCooperativeLaunchTooLarge`. That is
`max_active_blocks_per_sm × SMs` exactly — 6 × 170 — so the bound can be queried
with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` rather than discovered.

Note 6 blocks per SM here against sm90's 8, which follows from this part holding
48 warps per SM rather than 64.

**`[coop.ratio.dev.relaunch]` — a relaunch costs 1.40× a `grid_sync`.**

| | ns |
|---|---:|
| `grid_sync` at 132 blocks | 598.8 |
| device-side relaunch (64 launches in a CUDA graph) | 836.5 |

So a grid barrier buys ~29% over the relaunch it replaces, against sm90's 22%.
Still not an order of magnitude: a persistent kernel has to earn its keep on
state kept in registers and shared memory across phases, which this unit says
nothing about.

**A cross-check worth keeping.** The probe prints a hardcoded sm90 comparison —
"launch.lat.dev.ramp measured a launch at 1240 ns independently; this reads
837". Read against *this* machine's table, the 837 ns in-graph relaunch
corroborates `[launch.lat.dev.ramp]`'s **in-graph** figure of 450 ns as the
right order, and is nowhere near its in-stream 2050 ns. That agreement is what
caught the graph-versus-stream confusion in the launch unit.
