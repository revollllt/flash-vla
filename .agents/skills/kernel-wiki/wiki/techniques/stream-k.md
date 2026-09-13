---
id: technique-stream-k
title: "Stream-K: split only the partial wave, and give each fixup role to the CTA that arrives at the right time"
type: technique
architectures: [sm90, sm120]
tags: [stream-k, split-k, tile-scheduling, persistent-kernel]
confidence: source-reported
reproducibility: snippet
prerequisites: [technique-persistent-kernels]
related: [technique-tile-scheduling, technique-persistent-kernels, technique-reduction-own-task-kind, pattern-low-sm-utilization, pattern-tail-effect]
sources: [blog-humming, doc-cutlass-blackwell, doc-nvidia-tuning-guide]
blackwell_relevance: "Scheduling, not an ISA feature: humming carries one scheduler for sm90 and sm120 and differs only in the policy that selects it, so the mechanism ports where CLC and tcgen05 do not."
---

# Stream-K

Every schedule in [tile-scheduling](tile-scheduling.md) maps whole tiles, so
none of them helps when the tile count has no value near the SM count — which is
what powers-of-two dimensions give you. Stream-K stops tiling the output: each
CTA takes a contiguous slice of the flattened `(tile, k_chunk)` space, may begin
and end mid-tile, and the split tiles are reconciled afterwards.

The skeleton below is distilled from `inclusionAI/humming` — `scheduler.cuh`,
`epilogue/pipeline.cuh`, `utils/ptx/barrier.cuh` — which runs it unchanged on
sm90 and sm120.

## Host: plan the split

```cuda
// Occupancy, NOT sm_count. One CTA per SM throws away the latency hiding a
// multi-CTA-per-SM launch gets for free, and every CTA must be resident anyway
// because the fixup below spins.
int ctas = sm_count * ctas_per_sm;

int tiles = tiles_m * tiles_n;
int sk_tiles = 0;
if (tiles > ctas) {
  sk_tiles = tiles % ctas;                 // only the wave that does not fill
  // A remainder far smaller than the grid leaves each CTA too little work to
  // amortize its own fixup; borrow a whole wave and spread that instead.
  if (sk_tiles && sk_tiles * 10 <= ctas) sk_tiles += ctas;
}
int dp_tiles = tiles - sk_tiles;           // the bulk: whole tiles, no fixup

// One lock per split tile. Zero it ONCE: the release protocol restores it.
cudaMalloc(&locks, sk_tiles * sizeof(int));
cudaMemset(locks, 0, sk_tiles * sizeof(int));

gemm<<<ctas, threads>>>(..., locks, dp_tiles, sk_tiles, k_chunks);
```

## Device: two phases in one persistent kernel

```cuda
// Phase 1 -- data parallel. Whole tiles, whole K, output written directly.
for (int t = blockIdx.x; t < dp_tiles; t += gridDim.x) {
  float acc[kAccRegs] = {};
  mainloop(acc, tile_m_of(t), tile_n_of(t), /*k_begin=*/0, /*k_end=*/k_chunks);
  store(acc, t);
}

// Phase 2 -- stream-K over the remainder only. This CTA owns a contiguous
// slice of the (tile, k_chunk) iteration space, so it may enter a tile part
// way in and leave another part way out.
int total = sk_tiles * k_chunks;
int q = total / gridDim.x, r = total % gridDim.x;
int it  = blockIdx.x * q + min(blockIdx.x, r);
int end = it + q + (blockIdx.x < r ? 1 : 0);

while (it < end) {
  int tile = it / k_chunks;
  int k0   = it - tile * k_chunks;
  int k1   = min(k_chunks, end - tile * k_chunks);

  float acc[kAccRegs] = {};
  mainloop(acc, tile_m_of(dp_tiles + tile), tile_n_of(dp_tiles + tile), k0, k1);

  // Derived, not communicated: every CTA covering this tile computes the same
  // pair from the span length, so no CTA has to be told about the others.
  int slice_id, slice_count;
  slice_of(k0, k1, k_chunks, q, &slice_id, &slice_count);

  // REVERSE it. A CTA's slice covers the tail of one tile, then whole tiles,
  // then the head of another -- so the CTA holding a tile's LAST k-chunks
  // reaches it FIRST (the tile is where its slice begins) and the CTA holding
  // k-chunk 0 reaches it LAST. This fixup writes before it accumulates, so the
  // writer has to be the first arrival.
  //
  // Give either role to the wrong end and every CTA blocks at the head of its
  // own slice on a predecessor that has not started running: the grid
  // serializes into a chain.
  slice_id = slice_count - 1 - slice_id;

  fixup(acc, &locks[tile], slice_id, slice_count, tile);
  it = tile * k_chunks + k1;
}
```

`slice_of` is the one piece of bookkeeping worth transcribing exactly. Verbatim
from `get_streamk_next_block`, with `k_block_id` = `k0`, `slice_iters` =
`k1 - k0`, `K_BLOCKS` = `k_chunks`, and `streamk_mnk_total_iters` the per-CTA
span length `q`:

```cuda
  if (k_block_id == 0) {
    slice_id = 0;
    slice_count = CEIL_DIV(K_BLOCKS - slice_iters, streamk_mnk_total_iters) + 1;
  } else {
    slice_id = k_block_id / streamk_mnk_total_iters;
    uint32_t slice_first_block_iters =
        k_block_id - slice_id * streamk_mnk_total_iters;
    slice_count = CEIL_DIV(K_BLOCKS - slice_first_block_iters,
                           streamk_mnk_total_iters);
    if (slice_first_block_iters) {
      slice_id++;
      slice_count++;
    }
  }
```

## The fixup, and the lock that resets itself

```cuda
__device__ void fixup(const float *acc, int *lock, int slice_id,
                      int slice_count, int tile) {
  // The writer passes at once (lock == 0); accumulators wait for it to publish.
  barrier_acquire(lock, slice_id == 0 ? 0 : -1);

  if (slice_count == 1 || slice_id == 0) store(acc, tile);   // establishes
  else                                   atomic_add(acc, tile);  // accumulates

  // Writer publishes 1 - slice_count; each accumulator adds one. With four
  // slices the lock walks 0 -> -3 -> -2 -> -1 -> 0, so it ends where it began
  // and a graph replay or a following launch needs no memset.
  barrier_release(lock, slice_id == 0 ? 1 - slice_count : 0);
}

__device__ void barrier_acquire(int *lock, int count) {
  if (threadIdx.x == 0) {
    int state;
    do {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state) : "l"(lock) : "memory");
    } while (state > count);
  }
  __syncthreads();
}

__device__ void barrier_release(int *lock, int val) {
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");   // not __threadfence()
    if (val < 0) __stcg(lock, val);                      // writer publishes
    else asm volatile("red.relaxed.gpu.global.add.s32 [%0], 1;\n"
                      :: "l"(lock) : "memory");          // accumulator counts
  }
}
```

## Choosing the fixup family

| | accumulate into the output | partials + ordered reduce |
|---|---|---|
| extra buffer / traffic | none | one tile per split, written then read |
| reduction order | whichever CTA arrives | fixed |
| bitwise reproducible | **no** | yes |

The skeleton above is the first. It is why humming's batch-invariant mode does
not merely prefer another path — it asserts stream-K is off. **A consumer that
compares a replayed result against a previous one rather than against a
tolerance needs the second family**, which is what CUTLASS's stream-K swizzle
provides, at the cost of the round trip.

## What to measure

- Against the data-parallel form of the *same* kernel, so only the schedule moves.
- On a shape whose tile count is already a whole multiple of the grid: stream-K
  should be neutral there, and whatever it is not is fixup overhead.
- Thresholds — minimum K depth, minimum tile count, tile shape — are a search,
  not a constant to copy. humming keeps them in a per-architecture policy module
  for that reason.

## Caveat carried from the reading

`barrier_acquire`'s `state > count` releases every accumulator on the writer's
single store, so they are not serialized against each other. humming uses a
non-atomic read-modify-write when a tile has few slices, which would then race
for two accumulators; the headers read here do not show what resolves it. Settle
that before porting this family.
