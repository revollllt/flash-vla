# Category: memory — 访存

**Everything that moves bytes, and the descriptors and ordering that gate it.**
The copy engine, the atomic unit, the caches, and the address maps that decide
how many transactions a given access pattern becomes.

Arch-independent. Results live in `<arch>/unit-*.md`; this file is what to
measure and how to think about it when a new architecture arrives.

## Units in this category

| Unit | Probe | Measures |
|---|---|---|
| **tma** | `probes/memory/tma_ring.{cu,py}` | copy-engine delivery: per-warp issue rate, latency, ring depth, box geometry, descriptor limits, the saturation frontier, the per-CTA ceiling |
| **atomic** | `probes/memory/gmem_atomic.{cu,py}` | atomic throughput by instruction, width, scope, address placement and sharing; gmem-counter arrive→observe latency |

**Not yet a unit anywhere:** shared memory (`ldmatrix`/`stmatrix` bandwidth,
bank conflicts as a measured cost rather than a computed one), L2 as a unit in
its own right, TMA multicast, DSMEM, and cold-vs-cached atomics.

## The first question for any memory unit

**Is this path per-TRANSACTION or per-BYTE?** Ask it before anything else,
because the answer inverts the design rules: a per-transaction path is improved
by *fewer, larger* accesses and is indifferent to how many bytes each carries,
while a per-byte path is the opposite.

The decisive sweep is the same every time: **hold the operation count fixed and
vary the width.** If the op rate does not move, the path is per-transaction and
the extra bytes are free.

> On sm90 *both* measured units came back per-transaction, and neither was
> expected to. TMA: ~270 ns per box from 1 KB to the 32 KB descriptor maximum.
> Atomics: `u32` and `v4.f32` retire at the same rate, so the wide one moves
> 3.8× the bytes for free. **Treat this as a hypothesis to test on a new arch,
> not as a property of GPUs.**

## The second question: what does the address map cost?

Two separate sub-questions that are easy to conflate:

- **Geometry** — does a strided or gathered access cost more than a contiguous
  one at equal transaction count? (sm90 TMA: ≤3.4%. Nearly free.)
- **Coalescing** — does the hardware merge accesses from one warp that land in
  the same line, and how much is that worth? (sm90 atomics: **6.3×**. The
  largest single lever in the category.)

Geometry and coalescing pull in opposite directions under contention, so a
sweep that moves only one of them will mislead. Sweep the address *count* and
the address *spacing* independently, and read the two columns against each
other rather than down one.

## The third question: what saturates it, and what is the unit of saturation?

Every memory path has a concurrency knob — outstanding transactions, in-flight
bytes, requesting warps, requesting CTAs. Find which combination the hardware
actually counts. The test is an **iso-product sweep**: reach the same nominal
concurrency by different splits of the knobs, and see whether they agree.

> On sm90 the TMA answer was that delivery depends on the *product*
> `CTAs × warps × bytes-per-box` and on nothing else, to within the noise floor
> — but only up to a per-CTA ceiling, past which one CTA cannot absorb more
> however the product is reached. Both halves of that were found by iso-product
> sweeps; neither would have appeared in a one-axis-at-a-time sweep.

## Latency versus rate

Separate them explicitly, with a depth knob: **depth 1 measures latency, deeper
measures the issue rate.** Then the covering depth is `ceil(latency / interval)`
and it is a derived number, not a guess.

Do not infer latency from a partially-overlapped configuration. On sm90 an
earlier reading inferred TMA latency from a depth-2 row that was already
bandwidth-contaminated, and the recorded constant was wrong until a depth-1
sweep at low occupancy replaced it.

## Reporting rules specific to this category

- **Report against a measured bandwidth ceiling, never the datasheet peak.**
  Derive the ceiling from the same probe at maximum concurrency.
- **Say whether the source was cache-resident or cold.** They can be identical
  (sm90 TMA below 64 CTAs) or differ by 2.3× (the same unit above it), and a
  constant that does not say which was measured is unusable.
- **Compute the per-transaction interval by the shortest path from what you
  timed** — `time / transactions`, never via a byte count, so a bookkeeping
  error cannot masquerade as a hardware finding.
