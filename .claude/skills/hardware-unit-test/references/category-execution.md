# Category: execution — 执行与同步

**What it costs to start work and to order it.** Grid ramp, occupancy and
residency, thread-block clusters, barriers, and the counters kernels use to
sequence themselves. Nothing here moves useful bytes or does useful math; it is
all the overhead a design must budget around.

Arch-independent. Results live in `<arch>/unit-*.md`.

## Units in this category

| Unit | Probe | Measures |
|---|---|---|
| **launch** | none in this skill yet | per-launch grid ramp, cold streaming bandwidth against transfer size and CTA count, cluster declaration and barrier cost, cluster placement limits, occupancy versus residency |

The **gmem-counter hop** is an execution constant that is measured by the memory
category's atomic probe, because it is the same instruction path. It is recorded
in the atomic unit and cross-referenced here; do not measure it twice.

**Not yet a unit anywhere:** `__syncthreads` and named-barrier cost, mbarrier
`try_wait` polling cost, DSMEM push/pull, and what a spinning poller costs a
concurrent kernel.

## The questions an execution unit must answer

1. **What does a launch cost with an empty body**, at the real CTA count and
   the real shared-memory request? This is the floor every kernel starts in debt
   by, and it is often *grid ramp* rather than host overhead — check whether
   capturing the launch in a graph removes it. If it does not, **launch count
   becomes a first-class term in every fusion decision.**
2. **What is the reachable streaming bandwidth**, fitted across a range of
   transfer sizes rather than divided from one point? The fit separates a fixed
   cost from a marginal rate; a single point conflates them and the error is
   large at small sizes.
3. **How does that vary with grid size**, past the point where the grid covers
   the machine? More is not monotonically better.
4. **What do the placement and synchronisation primitives cost** — declaring a
   cluster, synchronising one, at each size?
5. **What does the occupancy query actually promise?** Capacity is not
   residency, and the placement of co-scheduled groups may follow the query
   rather than the physical SM count.

## Two traps this category exists to catch

**Capacity is not residency.** An occupancy query says how many blocks *could*
share an SM. The scheduler spreads a grid across SMs first, so a grid sized to
the SM count lands one block per SM whatever the query says, and freeing
resources to raise the query buys nothing until the grid grows to match. Verify
behaviourally — with a spin barrier that only completes if every block is
resident — not by reading the query.

**A synchronisation primitive's nominal cost is a floor, and placement
dominates it.** Measured in an empty kernel at zero skew, a barrier is cheap.
Inside a real kernel the same barrier costs whatever skew it absorbs, and moving
it changes that: at the *end* of a kernel it absorbs skew already being paid, at
the *start* it adds fill skew to the critical path. **Design the number of
barriers, not the size of the group.**

## Reporting rules specific to this category

- **State whether the measurement was inside or outside a graph capture**, and
  which timer was used. At the few-microsecond scale the timer's own overhead is
  most of the measurement.
- **Costs here are per launch, per barrier, per hop** — countable things. Record
  them that way, so a design can multiply rather than interpolate.
- **Broadcast and fan-out deserve their own sweep.** Whether an ordering
  primitive's cost grows with the number of observers changes the shape of a
  task graph, and it is cheap to measure.
