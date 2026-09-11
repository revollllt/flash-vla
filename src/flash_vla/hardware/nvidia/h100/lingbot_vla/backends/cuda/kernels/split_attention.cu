// Split-key masked GQA attention for the LingBot action expert, in two launches.
//
// The expert's attention is 16 query heads over 51 suffix rows and 315 cached
// keys of 128 float32 dimensions, run 360 times per forward. As three launches
// -- cuBLAS QK, a masked softmax, cuBLAS PV -- it costs 18.5 us per layer-step
// (7.08 + 4.56 + 6.85, job 614367) against a ~2.4 us streaming floor for the
// 1.5 MB it touches and a ~2.2 us float32 FMA roofline for its 66 M MACs.
//
// The earlier one-launch flash kernel gave one warp to each (head, query row)
// and had all 816 of them stream the whole 645 KB cache: half a gigabyte of L2
// traffic per layer-step, and 7x slower end to end than the chain it replaced.
// This kernel inverts that. The key axis is cut into `key_tile` slices and the
// grid is (heads x slices x row tiles), so a CTA reads one slice of the cache
// exactly once, stages it in shared memory, and every query row it owns
// consumes it from there. At key_tile = 40 and one row tile that is 16 x 8 =
// 128 CTAs, one wave on the 132 SMs, and the global traffic is ~8 MB of
// L2-resident reads rather than ~500 MB.
//
// Each CTA emits a flash-decoding partial -- a running maximum, a denominator
// and an unnormalized 128-dim accumulator per (head, row) -- and the second
// kernel rescales the slices of a row onto their common maximum and writes the
// transposed bf16 result the output projection consumes. The score tensor never
// reaches memory in either launch.
//
// Both mainloops are register-blocked so the shared-memory reads are amortized
// rather than one load per MAC, and neither contains a warp reduction: the QK
// pass gives each lane whole rows and each warp whole keys, so a dot product is
// accumulated in one lane's registers over the 128-dimension axis, and the PV
// pass gives each lane four output dimensions, which is already a per-lane
// accumulation over keys. The cross-warp maximum and denominator are two
// 8-element shared-memory reductions per row for the whole 315-key row, not one
// per key.
//
// The grid is too small to fill the machine more than once -- 128 CTAs of
// eight warps is 1024 warps where an H100 holds 8448 -- so a CTA has almost no
// other work to hide its own latency behind, and that, not throughput, is what
// the shape parameters are fighting. Measured with the `kTimed` instrumentation
// below (jobs 614654 / 614692, key_tile 40, ~1.7 GHz), per CTA:
//   - Staging cost 6650 of 18771 cycles when each thread waited on its own
//     global load before storing it to shared memory. `cp.async` puts every
//     copy a thread owns in flight at once and brought it to 4211, which is
//     ~3.5 TB/s for the grid's 8.8 MB and so bandwidth-bound: the remaining
//     lever would be reading less, and the decomposition fixes that volume.
//   - The QK mainloop runs at ~2.3x and the PV mainloop at ~1.6x their
//     instruction-issue bounds. Neither more warps per SM (`row_tile` 32, two
//     CTAs per SM: 10.23 us against 9.48) nor half the wide shared-memory reads
//     (`row_groups` 2, which trades them for broadcasts: 10.11 against 9.27)
//     improves on the plain shape, so the cost is scheduler-level latency this
//     grid cannot cover, not bandwidth and not code quality -- the SASS inner
//     loops are exactly 80 FFMA + 14 LDS.128 and 224 FFMA + 22 LDS.128. Both
//     alternatives stay instantiated so the negative result is re-runnable.
//   - `pieces` gives the combine's 816 warps of work a second warp per output
//     row, which is worth ~0.2 us of its ~2.8.
//   - The TF32 tensor-core mainloops below are 1.5x and 1.3x faster than the
//     float32 ones per CTA (QK 7229 -> 4671 cycles, PV 4126 -> 3064) and buy
//     nothing end to end: 12.57 us against 12.53 (job 615519). The math was
//     never the constraint. What is left is staging bandwidth, the combine's
//     partial traffic, and 1.76 us of two graph nodes -- about two thirds of
//     the total, none of it arithmetic.
//
// Arithmetic is float32 throughout, as the upstream eager path's is. Masked
// logits take upstream's finite sentinel rather than -inf, so a fully masked
// row still yields a uniform distribution -- including across the split, where
// every slice reports the sentinel as its maximum and its own live key count as
// its denominator, and the combine sums them to the full 315.
//
// Layouts are the ones the projection epilogue already writes: `query` is
// [heads, rows, dim] contiguous, the caches are [kv_heads, keys, dim] with
// caller-supplied head and key strides (the backbone's caches are a transposed
// view of the graph's [keys, kv_heads, dim] buffers), and only the `dim` axis
// has to be contiguous. The output is [rows, heads * dim] bf16.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarps = 8;                      // 256 threads: 8 warps over 4
                                               // schedulers keeps two FFMA
                                               // streams per scheduler in flight
constexpr int kThreads = kWarpSize * kWarps;
constexpr int kDim = 128;                      // this Target's head dimension
constexpr float kMaskedLogit = -2.3819763e38f;

// Row stride of the shared query tile. The QK mainloop reads it as float4 at a
// lane-dependent row, so the eight lanes of a 128-bit access phase must land on
// eight distinct 16-byte banks: 132 floats is 528 bytes = 33 banks, and 33 is
// odd, so lane l lands on bank l % 8. A stride of 128 would put all 32 lanes on
// one bank.
constexpr int kQueryStride = kDim + 4;
constexpr int kQuads = kDim / 4;

__device__ __forceinline__ float4 load4(const float* address) {
    return *reinterpret_cast<const float4*>(address);
}

__device__ __forceinline__ void store4(float* address, const float4& value) {
    *reinterpret_cast<float4*>(address) = value;
}

// One 16-byte global-to-shared copy, issued and not waited on. The whole
// staging phase is issued before any of it is consumed, so the phase costs
// bandwidth rather than one global latency per thread per tile row.
__device__ __forceinline__ void copy16(float* destination, const float* source) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address), "l"(source));
}

__device__ __forceinline__ void copy_wait() {
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
}

struct __align__(8) BFloat4 {
    __nv_bfloat162 low;
    __nv_bfloat162 high;
};

// Shared-memory budget for one CTA, in bytes. The caller raises the dynamic
// shared-memory limit to this before the first launch; at key_tile = 40 and two
// row slots it is 92.7 KB, inside the 227 KB an H100 CTA may opt into, and at
// one row slot it is 65.8 KB, so two CTAs share an SM.
constexpr int shared_bytes(int key_tile, int row_tile, int key_groups) {
    return (int)sizeof(float) * (row_tile * kQueryStride       // query tile
                                 + 2 * key_tile * kDim         // key and value tiles
                                 + row_tile * (key_tile + 4)   // probability tile
                                 + 2 * key_groups * row_tile)  // the two reductions
           + key_tile * row_tile;                              // mask tile
}

// One key slice of one row tile of one head: rows x key_tile scores, softmaxed
// against this slice's own maximum, applied to the slice's values.
//
// Thread roles. threadIdx.x is the lane and threadIdx.y the warp.
//   QK mainloop: lane -> query rows {lane, lane + 32, ...}, warp ->
//     kKeysPerWarp consecutive keys. Each thread holds a kRowSlots x
//     kKeysPerWarp score tile in registers and reduces over `dim` locally, so
//     there is no shuffle.
//   PV mainloop: warp -> kRowsPerWarp consecutive query rows, lane -> output
//     dimensions {4 * lane .. 4 * lane + 3}. Each thread holds a
//     kRowsPerWarp x 4 accumulator tile and reduces over keys locally.
// The two roles disagree, so the probabilities pass through shared memory; the
// CTA barrier between them also publishes the two cross-warp reductions.
//
// `kTimed` adds four SM-clock reads at the phase boundaries and has one thread
// per CTA publish the deltas. It is a separate instantiation, so the shipped
// one carries neither the reads nor the branch.
template <int kKeysPerWarp, int kRowGroups, int kRowTile, int kRowsPerWarp, int kUnroll = 2,
          bool kTimed = false>
__global__ __launch_bounds__(kThreads) void split_attention_kernel(
    const float* __restrict__ query, const float* __restrict__ keys,
    const float* __restrict__ values, const bool* __restrict__ mask,
    float* __restrict__ partial, float* __restrict__ partial_max,
    float* __restrict__ partial_sum, long long* __restrict__ timing,
    int rows, int heads, int group, int key_count,
    long long head_stride, long long key_stride, float scale) {
    const long long entered = kTimed ? clock64() : 0;
    // The eight warps form a kRowGroups x kKeyGroups grid over the score tile.
    // Every warp reads the whole of whichever operand its group does not
    // subdivide, so kRowGroups = 2 halves the wide query reads each warp makes
    // and doubles its broadcast key reads -- a shared-memory bandwidth against
    // instruction-count trade that only measurement settles.
    constexpr int kKeyGroups = kWarps / kRowGroups;
    constexpr int kKeyTile = kKeysPerWarp * kKeyGroups;
    constexpr int kRowSlots = kRowTile / (kWarpSize * kRowGroups);
    constexpr int kMaxRows = kRowTile;
    // Multiple of four so the PV mainloop can read four probabilities as one
    // float4; 16-byte aligned rows cost at most a four-way store conflict in
    // the QK epilogue, which is ten stores against ~2700 mainloop cycles.
    constexpr int kProbStride = kKeyTile + 4;
    constexpr int kQueryCopies = (kMaxRows * kQuads + kThreads - 1) / kThreads;
    constexpr int kCacheCopies = (kKeyTile * kQuads + kThreads - 1) / kThreads;
    constexpr int kMaskLoads = (kMaxRows * kKeyTile + kThreads - 1) / kThreads;

    extern __shared__ __align__(16) float smem[];
    float* query_tile = smem;
    float* key_tile = query_tile + kMaxRows * kQueryStride;
    float* value_tile = key_tile + kKeyTile * kDim;
    float* prob_tile = value_tile + kKeyTile * kDim;
    float* reduce_max = prob_tile + kMaxRows * kProbStride;
    float* reduce_sum = reduce_max + kKeyGroups * kMaxRows;
    bool* mask_tile = reinterpret_cast<bool*>(reduce_sum + kKeyGroups * kMaxRows);

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int thread = warp * kWarpSize + lane;
    const int row_group = warp / kKeyGroups;
    const int key_group = warp - row_group * kKeyGroups;
    const int head = blockIdx.x;
    const int first_key = blockIdx.y * kKeyTile;
    const int first_row = blockIdx.z * kMaxRows;
    const int active = min(kKeyTile, key_count - first_key);
    const int live = min(kMaxRows, rows - first_row);

    const int kv = head / group;
    const float* key_source = keys + kv * head_stride + (long long)first_key * key_stride;
    const float* value_source = values + kv * head_stride + (long long)first_key * key_stride;
    const float* query_source = query + ((long long)head * rows + first_row) * kDim;

    // Stage the tiles. Padding rows and keys are zeroed rather than left
    // undefined: the QK epilogue forces their logits to the sentinel and the PV
    // mainloop multiplies their values by a zero probability, and a NaN left in
    // shared memory would survive both.
#pragma unroll
    for (int i = 0; i < kQueryCopies; ++i) {
        const int index = thread + i * kThreads;
        if (index < kMaxRows * kQuads) {
            const int row = index / kQuads;
            const int quad = index - row * kQuads;
            float* destination = query_tile + row * kQueryStride + quad * 4;
            if (row < live) {
                copy16(destination, query_source + (long long)row * kDim + quad * 4);
            } else {
                store4(destination, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kCacheCopies; ++i) {
        const int index = thread + i * kThreads;
        if (index < kKeyTile * kQuads) {
            const int key = index / kQuads;
            const int quad = index - key * kQuads;
            if (key < active) {
                copy16(key_tile + key * kDim + quad * 4,
                       key_source + key * key_stride + quad * 4);
                copy16(value_tile + key * kDim + quad * 4,
                       value_source + key * key_stride + quad * 4);
            } else {
                store4(key_tile + key * kDim + quad * 4, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
                store4(value_tile + key * kDim + quad * 4, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
            }
        }
    }
    // The mask is bytes, so it cannot ride cp.async; buffering the loads in
    // registers first keeps them in flight together, and they overlap the tile
    // copies above. Read row-major so the global loads coalesce, write
    // key-major so the QK epilogue's lane-indexed reads do. Keys past `active`
    // are marked masked, which is what forces their logit to the sentinel.
    bool held[kMaskLoads];
#pragma unroll
    for (int i = 0; i < kMaskLoads; ++i) {
        const int index = thread + i * kThreads;
        const int row = index / kKeyTile;
        const int key = index - row * kKeyTile;
        held[i] = index < kMaxRows * kKeyTile && row < live && key < active
                  && mask[(long long)(first_row + row) * key_count + first_key + key];
    }
#pragma unroll
    for (int i = 0; i < kMaskLoads; ++i) {
        const int index = thread + i * kThreads;
        if (index < kMaxRows * kKeyTile) {
            const int row = index / kKeyTile;
            mask_tile[(index - row * kKeyTile) * kMaxRows + row] = held[i];
        }
    }
    copy_wait();
    __syncthreads();
    const long long staged = kTimed ? clock64() : 0;

    float score[kRowSlots][kKeysPerWarp];
#pragma unroll
    for (int r = 0; r < kRowSlots; ++r) {
#pragma unroll
        for (int k = 0; k < kKeysPerWarp; ++k) {
            score[r][k] = 0.0f;
        }
    }
    const int lane_row = row_group * kRowSlots * kWarpSize + lane;
    const int warp_key = key_group * kKeysPerWarp;
    const float* query_lane = query_tile + lane_row * kQueryStride;
    const float* key_warp = key_tile + warp_key * kDim;
#pragma unroll kUnroll
    for (int d = 0; d < kDim; d += 4) {
        float4 q[kRowSlots];
#pragma unroll
        for (int r = 0; r < kRowSlots; ++r) {
            q[r] = load4(query_lane + r * kWarpSize * kQueryStride + d);
        }
        float4 k[kKeysPerWarp];
#pragma unroll
        for (int j = 0; j < kKeysPerWarp; ++j) {
            k[j] = load4(key_warp + j * kDim + d);
        }
#pragma unroll
        for (int r = 0; r < kRowSlots; ++r) {
#pragma unroll
            for (int j = 0; j < kKeysPerWarp; ++j) {
                score[r][j] = fmaf(q[r].x, k[j].x, score[r][j]);
                score[r][j] = fmaf(q[r].y, k[j].y, score[r][j]);
                score[r][j] = fmaf(q[r].z, k[j].z, score[r][j]);
                score[r][j] = fmaf(q[r].w, k[j].w, score[r][j]);
            }
        }
    }

    float logit[kRowSlots][kKeysPerWarp];
#pragma unroll
    for (int r = 0; r < kRowSlots; ++r) {
        const int row = lane_row + r * kWarpSize;
        float best = kMaskedLogit;
#pragma unroll
        for (int j = 0; j < kKeysPerWarp; ++j) {
            logit[r][j] = mask_tile[(warp_key + j) * kMaxRows + row] ? score[r][j] * scale
                                                                     : kMaskedLogit;
            best = fmaxf(best, logit[r][j]);
        }
        reduce_max[key_group * kMaxRows + row] = best;
    }
    __syncthreads();

#pragma unroll
    for (int r = 0; r < kRowSlots; ++r) {
        const int row = lane_row + r * kWarpSize;
        float best = reduce_max[row];
#pragma unroll
        for (int w = 1; w < kKeyGroups; ++w) {
            best = fmaxf(best, reduce_max[w * kMaxRows + row]);
        }
        float total = 0.0f;
#pragma unroll
        for (int j = 0; j < kKeysPerWarp; ++j) {
            const int key = warp_key + j;
            // Past `active` the probability has to be a hard zero, not
            // exp(sentinel - sentinel) = 1: a fully masked row has the sentinel
            // as its maximum, and a padded key would then join its denominator.
            const float probability = key < active ? __expf(logit[r][j] - best) : 0.0f;
            total += probability;
            prob_tile[row * kProbStride + key] = probability;
        }
        reduce_sum[key_group * kMaxRows + row] = total;
    }
    __syncthreads();
    const long long scored = kTimed ? clock64() : 0;

    // Warp 0 owns publishing the per-row statistics, so the eight-warp
    // reduction happens once per row rather than once per row per warp.
    const int splits = gridDim.y;
    const long long slice = ((long long)head * rows + first_row) * splits + blockIdx.y;
    if (warp == 0) {
#pragma unroll
        for (int r = 0; r < kRowTile / kWarpSize; ++r) {
            const int row = lane + r * kWarpSize;
            if (row < live) {
                float best = reduce_max[row];
                float total = reduce_sum[row];
#pragma unroll
                for (int w = 1; w < kKeyGroups; ++w) {
                    best = fmaxf(best, reduce_max[w * kMaxRows + row]);
                    total += reduce_sum[w * kMaxRows + row];
                }
                partial_max[slice + (long long)row * splits] = best;
                partial_sum[slice + (long long)row * splits] = total;
            }
        }
    }

    float accumulator[kRowsPerWarp][4];
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            accumulator[i][e] = 0.0f;
        }
    }
    const int warp_row = warp * kRowsPerWarp;
    const float* value_lane = value_tile + lane * 4;
    const float* prob_rows = prob_tile + warp_row * kProbStride;
#pragma unroll 2
    for (int key = 0; key < kKeyTile; key += 4) {
        float4 v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] = load4(value_lane + (key + j) * kDim);
        }
#pragma unroll
        for (int i = 0; i < kRowsPerWarp; ++i) {
            const float4 p = load4(prob_rows + i * kProbStride + key);
            accumulator[i][0] = fmaf(p.x, v[0].x, accumulator[i][0]);
            accumulator[i][1] = fmaf(p.x, v[0].y, accumulator[i][1]);
            accumulator[i][2] = fmaf(p.x, v[0].z, accumulator[i][2]);
            accumulator[i][3] = fmaf(p.x, v[0].w, accumulator[i][3]);
            accumulator[i][0] = fmaf(p.y, v[1].x, accumulator[i][0]);
            accumulator[i][1] = fmaf(p.y, v[1].y, accumulator[i][1]);
            accumulator[i][2] = fmaf(p.y, v[1].z, accumulator[i][2]);
            accumulator[i][3] = fmaf(p.y, v[1].w, accumulator[i][3]);
            accumulator[i][0] = fmaf(p.z, v[2].x, accumulator[i][0]);
            accumulator[i][1] = fmaf(p.z, v[2].y, accumulator[i][1]);
            accumulator[i][2] = fmaf(p.z, v[2].z, accumulator[i][2]);
            accumulator[i][3] = fmaf(p.z, v[2].w, accumulator[i][3]);
            accumulator[i][0] = fmaf(p.w, v[3].x, accumulator[i][0]);
            accumulator[i][1] = fmaf(p.w, v[3].y, accumulator[i][1]);
            accumulator[i][2] = fmaf(p.w, v[3].z, accumulator[i][2]);
            accumulator[i][3] = fmaf(p.w, v[3].w, accumulator[i][3]);
        }
    }

    const long long weighted = kTimed ? clock64() : 0;
    float* out = partial + slice * kDim;
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
        const int row = warp_row + i;
        if (row >= live) {
            break;
        }
        store4(out + (long long)row * splits * kDim + lane * 4,
               make_float4(accumulator[i][0], accumulator[i][1],
                           accumulator[i][2], accumulator[i][3]));
    }

    if (kTimed && thread == 0) {
        long long* record = timing + 4 * (blockIdx.y * gridDim.x + head);
        record[0] = staged - entered;      // global -> shared staging
        record[1] = scored - staged;       // QK mainloop and its softmax epilogue
        record[2] = weighted - scored;     // PV mainloop
        record[3] = clock64() - entered;   // whole CTA
    }
}

// ---------------------------------------------------------------------------
// TF32 tensor-core variant of the slices kernel.
//
// The upstream eager attention this is validated against already runs its two
// products on TF32 tensor cores -- the baseline profile of the unmodified route
// (job 614222) shows `sm80_xmma_gemm_f32f32_tf32f32_f32` for QK and
// `cutlass_80_tensorop_s1688gemm` (m16n8k8) for PV -- so a TF32 mainloop moves
// toward the reference's numerics rather than away from them. Inputs are
// rounded with `cvt.rna.tf32.f32`, which is what cuBLAS does, and every
// accumulation stays float32.
//
// Both products map onto `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`
// without transposing anything in shared memory. The instruction fixes which
// (m, k) and (k, n) element each lane holds, and this kernel assembles those
// fragments by direct shared-memory reads, so the caches can stay in their
// natural [key][dim] order for both passes:
//   QK: A = query[row][dim] (m, k), B[k][n] = key[n = key][k = dim].
//   PV: A = probability[row][key] (m, k), B[k][n] = value[k = key][n = dim].
// The eight warps are four 16-row blocks by two column groups: two key groups
// in QK, two 64-wide dimension groups in PV.
//
// Every shared-memory stride is chosen against a fragment's access pattern, not
// against a vector width. A fragment lane holds lane/4 as the row and lane%4 as
// the k offset, so the 32 lanes land on 32 distinct banks when the row stride
// is 4 mod 32 words (query, key, probability); the B fragment of PV transposes
// those roles, so its stride wants 8 mod 32 (value).
//
// The mask becomes one 64-bit word per query row, built with `__ballot_sync`
// over coalesced reads at staging time. Read as a byte per (row, key) in the
// fragment layout it is a 16-way bank conflict on twelve loads per thread,
// which would cost more than the mainloop it guards; as a bit test it is two
// broadcast loads for the whole epilogue.
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned to_tf32(float value) {
    unsigned rounded;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(rounded) : "f"(value));
    return rounded;
}

__device__ __forceinline__ void mma_tf32(float (&accumulator)[4], const unsigned (&a)[4],
                                         const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]),
          "+f"(accumulator[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Row stride of the shared probability tile: the smallest width at or above the
// key tile that is 4 mod 32 words, so PV's A fragment is conflict-free.
__host__ __device__ constexpr int prob_stride(int key_tile) {
    int stride = key_tile;
    while (stride % 32 != 4) {
        ++stride;
    }
    return stride;
}

constexpr int kTensorRows = 64;             // four 16-row MMA blocks
constexpr int kTensorQueryStride = kDim + 4;    // 4 mod 32: A fragment of QK
constexpr int kTensorKeyStride = kDim + 4;      // 4 mod 32: B fragment of QK
constexpr int kTensorValueStride = kDim + 8;    // 8 mod 32: B fragment of PV

constexpr int tensor_shared_bytes(int key_tile) {
    return (int)sizeof(float) * (kTensorRows * kTensorQueryStride
                                 + key_tile * kTensorKeyStride
                                 + key_tile * kTensorValueStride
                                 + kTensorRows * prob_stride(key_tile)
                                 + 2 * 2 * kTensorRows)
           + (int)sizeof(unsigned long long) * kTensorRows;
}

template <int kKeyTile, bool kTimed = false>
__global__ __launch_bounds__(kThreads) void split_attention_tensor_kernel(
    const float* __restrict__ query, const float* __restrict__ keys,
    const float* __restrict__ values, const bool* __restrict__ mask,
    float* __restrict__ partial, float* __restrict__ partial_max,
    float* __restrict__ partial_sum, long long* __restrict__ timing,
    int rows, int heads, int group, int key_count,
    long long head_stride, long long key_stride, float scale) {
    const long long entered = kTimed ? clock64() : 0;
    constexpr int kProbStride = prob_stride(kKeyTile);
    constexpr int kKeysPerGroup = kKeyTile / 2;
    constexpr int kQkTiles = kKeysPerGroup / 8;      // n tiles of the QK product
    constexpr int kPvTiles = kDim / 2 / 8;           // n tiles of the PV product
    constexpr int kQuads = kDim / 4;
    constexpr int kQueryCopies = (kTensorRows * kQuads + kThreads - 1) / kThreads;
    constexpr int kCacheCopies = (kKeyTile * kQuads + kThreads - 1) / kThreads;
    constexpr int kMaskRows = kTensorRows / kWarps;  // rows one warp ballots for

    extern __shared__ __align__(16) float smem[];
    float* query_tile = smem;
    float* key_tile = query_tile + kTensorRows * kTensorQueryStride;
    float* value_tile = key_tile + kKeyTile * kTensorKeyStride;
    float* prob_tile = value_tile + kKeyTile * kTensorValueStride;
    float* reduce_max = prob_tile + kTensorRows * kProbStride;
    float* reduce_sum = reduce_max + 2 * kTensorRows;
    unsigned long long* mask_bits =
        reinterpret_cast<unsigned long long*>(reduce_sum + 2 * kTensorRows);

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int thread = warp * kWarpSize + lane;
    const int block_row = (warp >> 1) * 16;          // this warp's 16-row block
    const int column_group = warp & 1;               // keys in QK, dimensions in PV
    const int lane_row = lane >> 2;                  // MMA fragment row within eight
    const int lane_step = lane & 3;                  // MMA fragment k offset

    const int head = blockIdx.x;
    const int split = blockIdx.y;
    const int first_key = split * kKeyTile;
    const int first_row = blockIdx.z * kTensorRows;
    const int active = min(kKeyTile, key_count - first_key);
    const int live = min(kTensorRows, rows - first_row);

    const int kv = head / group;
    const float* key_source = keys + kv * head_stride + (long long)first_key * key_stride;
    const float* value_source = values + kv * head_stride + (long long)first_key * key_stride;
    const float* query_source = query + ((long long)head * rows + first_row) * kDim;

    // Staging, as in the float32 kernel: every copy a thread owns is in flight
    // before any of it is consumed, and padding rows and keys are zeroed so a
    // NaN cannot survive into a product whose weight is zero.
#pragma unroll
    for (int i = 0; i < kQueryCopies; ++i) {
        const int index = thread + i * kThreads;
        if (index < kTensorRows * kQuads) {
            const int row = index / kQuads;
            const int quad = index - row * kQuads;
            float* destination = query_tile + row * kTensorQueryStride + quad * 4;
            if (row < live) {
                copy16(destination, query_source + (long long)row * kDim + quad * 4);
            } else {
                store4(destination, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kCacheCopies; ++i) {
        const int index = thread + i * kThreads;
        if (index < kKeyTile * kQuads) {
            const int key = index / kQuads;
            const int quad = index - key * kQuads;
            float* to_key = key_tile + key * kTensorKeyStride + quad * 4;
            float* to_value = value_tile + key * kTensorValueStride + quad * 4;
            if (key < active) {
                copy16(to_key, key_source + key * key_stride + quad * 4);
                copy16(to_value, value_source + key * key_stride + quad * 4);
            } else {
                store4(to_key, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
                store4(to_value, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
            }
        }
    }
    // One 64-bit mask word per query row, so the epilogue tests bits instead of
    // gathering bytes. Each warp ballots for its own eight rows over coalesced
    // reads; kKeyTile <= 64 is what makes one word enough.
#pragma unroll
    for (int i = 0; i < kMaskRows; ++i) {
        const int row = warp * kMaskRows + i;
        const bool usable = row < live;
        const bool low = usable && lane < active
                         && mask[(long long)(first_row + row) * key_count + first_key + lane];
        const unsigned low_bits = __ballot_sync(0xffffffffu, low);
        unsigned high_bits = 0;
        if (kKeyTile > 32) {
            const int key = lane + 32;
            const bool high = usable && key < active
                              && mask[(long long)(first_row + row) * key_count + first_key
                                      + key];
            high_bits = __ballot_sync(0xffffffffu, high);
        }
        if (lane == 0) {
            mask_bits[row] = (unsigned long long)low_bits
                             | ((unsigned long long)high_bits << 32);
        }
    }
    copy_wait();
    __syncthreads();
    const long long staged = kTimed ? clock64() : 0;

    // QK: this warp's 16 rows against its key group, one m16n8k8 per n tile per
    // eight dimensions.
    float score[kQkTiles][4];
#pragma unroll
    for (int j = 0; j < kQkTiles; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            score[j][e] = 0.0f;
        }
    }
    const int key_base = column_group * kKeysPerGroup;
    const float* query_fragment = query_tile + (block_row + lane_row) * kTensorQueryStride
                                  + lane_step;
    const float* key_fragment = key_tile + (key_base + lane_row) * kTensorKeyStride + lane_step;
#pragma unroll 2
    for (int step = 0; step < kDim; step += 8) {
        unsigned a[4];
        a[0] = to_tf32(query_fragment[step]);
        a[1] = to_tf32(query_fragment[8 * kTensorQueryStride + step]);
        a[2] = to_tf32(query_fragment[step + 4]);
        a[3] = to_tf32(query_fragment[8 * kTensorQueryStride + step + 4]);
#pragma unroll
        for (int j = 0; j < kQkTiles; ++j) {
            unsigned b[2];
            b[0] = to_tf32(key_fragment[8 * j * kTensorKeyStride + step]);
            b[1] = to_tf32(key_fragment[8 * j * kTensorKeyStride + step + 4]);
            mma_tf32(score[j], a, b);
        }
    }

    // Softmax epilogue in the accumulator's own layout: lane (4 * g + t) holds
    // rows g and g + 8 of the block at key columns 2t and 2t + 1 of each n tile,
    // so a row's eight columns live in four consecutive lanes and its maximum
    // needs two shuffles rather than a shared-memory round trip.
    const int row_low = block_row + lane_row;
    const int row_high = row_low + 8;
    const unsigned long long bits_low = mask_bits[row_low];
    const unsigned long long bits_high = mask_bits[row_high];
    float logit[kQkTiles][4];
    float best_low = kMaskedLogit;
    float best_high = kMaskedLogit;
#pragma unroll
    for (int j = 0; j < kQkTiles; ++j) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int key = key_base + 8 * j + 2 * lane_step + c;
            logit[j][c] = ((bits_low >> key) & 1ull) ? score[j][c] * scale : kMaskedLogit;
            logit[j][2 + c] = ((bits_high >> key) & 1ull) ? score[j][2 + c] * scale
                                                          : kMaskedLogit;
            best_low = fmaxf(best_low, logit[j][c]);
            best_high = fmaxf(best_high, logit[j][2 + c]);
        }
    }
#pragma unroll
    for (int offset = 1; offset < 4; offset <<= 1) {
        best_low = fmaxf(best_low, __shfl_xor_sync(0xffffffffu, best_low, offset));
        best_high = fmaxf(best_high, __shfl_xor_sync(0xffffffffu, best_high, offset));
    }
    if (lane_step == 0) {
        reduce_max[column_group * kTensorRows + row_low] = best_low;
        reduce_max[column_group * kTensorRows + row_high] = best_high;
    }
    __syncthreads();

    best_low = fmaxf(reduce_max[row_low], reduce_max[kTensorRows + row_low]);
    best_high = fmaxf(reduce_max[row_high], reduce_max[kTensorRows + row_high]);
    float total_low = 0.0f;
    float total_high = 0.0f;
    float* prob_low = prob_tile + row_low * kProbStride + key_base + 2 * lane_step;
    float* prob_high = prob_tile + row_high * kProbStride + key_base + 2 * lane_step;
#pragma unroll
    for (int j = 0; j < kQkTiles; ++j) {
        float low[2];
        float high[2];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int key = key_base + 8 * j + 2 * lane_step + c;
            // Past `active` the probability has to be a hard zero, not
            // exp(sentinel - sentinel) = 1: a fully masked row has the sentinel
            // as its maximum, and a padded key would then join its denominator.
            low[c] = key < active ? __expf(logit[j][c] - best_low) : 0.0f;
            high[c] = key < active ? __expf(logit[j][2 + c] - best_high) : 0.0f;
            total_low += low[c];
            total_high += high[c];
        }
        *reinterpret_cast<float2*>(prob_low + 8 * j) = make_float2(low[0], low[1]);
        *reinterpret_cast<float2*>(prob_high + 8 * j) = make_float2(high[0], high[1]);
    }
#pragma unroll
    for (int offset = 1; offset < 4; offset <<= 1) {
        total_low += __shfl_xor_sync(0xffffffffu, total_low, offset);
        total_high += __shfl_xor_sync(0xffffffffu, total_high, offset);
    }
    if (lane_step == 0) {
        reduce_sum[column_group * kTensorRows + row_low] = total_low;
        reduce_sum[column_group * kTensorRows + row_high] = total_high;
    }
    __syncthreads();
    const long long scored = kTimed ? clock64() : 0;

    const int splits = gridDim.y;
    const long long slice = ((long long)head * rows + first_row) * splits + split;
    if (warp == 0) {
#pragma unroll
        for (int r = 0; r < kTensorRows / kWarpSize; ++r) {
            const int row = lane + r * kWarpSize;
            if (row < live) {
                partial_max[slice + (long long)row * splits] =
                    fmaxf(reduce_max[row], reduce_max[kTensorRows + row]);
                partial_sum[slice + (long long)row * splits] =
                    reduce_sum[row] + reduce_sum[kTensorRows + row];
            }
        }
    }

    // PV: the same 16 rows against this warp's half of the output dimensions.
    float context[kPvTiles][4];
#pragma unroll
    for (int j = 0; j < kPvTiles; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            context[j][e] = 0.0f;
        }
    }
    const int dim_base = column_group * (kDim / 2);
    const float* prob_fragment = prob_tile + (block_row + lane_row) * kProbStride + lane_step;
    const float* value_fragment = value_tile + lane_step * kTensorValueStride + dim_base
                                  + lane_row;
#pragma unroll 2
    for (int step = 0; step < kKeyTile; step += 8) {
        unsigned a[4];
        a[0] = to_tf32(prob_fragment[step]);
        a[1] = to_tf32(prob_fragment[8 * kProbStride + step]);
        a[2] = to_tf32(prob_fragment[step + 4]);
        a[3] = to_tf32(prob_fragment[8 * kProbStride + step + 4]);
        const float* value_step = value_fragment + step * kTensorValueStride;
#pragma unroll
        for (int j = 0; j < kPvTiles; ++j) {
            unsigned b[2];
            b[0] = to_tf32(value_step[8 * j]);
            b[1] = to_tf32(value_step[4 * kTensorValueStride + 8 * j]);
            mma_tf32(context[j], a, b);
        }
    }

    // The MMA accumulator holds eight rows per warp at a two-wide column
    // stride, so storing it straight to global memory writes sixteen 32-byte
    // chunks per thread scattered across eight rows -- measured at 3499 of the
    // CTA's cycles against 432 for the float32 kernel's contiguous stores (job
    // 615490), which was the whole of the tensor mainloops' win. Land it in the
    // query tile instead, which has been dead since the QK barrier, and let
    // every warp make one coalesced pass over it.
    float* out_tile = query_tile;
#pragma unroll
    for (int j = 0; j < kPvTiles; ++j) {
        const int dimension = dim_base + 8 * j + 2 * lane_step;
        *reinterpret_cast<float2*>(out_tile + row_low * kTensorQueryStride + dimension) =
            make_float2(context[j][0], context[j][1]);
        *reinterpret_cast<float2*>(out_tile + row_high * kTensorQueryStride + dimension) =
            make_float2(context[j][2], context[j][3]);
    }
    __syncthreads();

    const long long weighted = kTimed ? clock64() : 0;
    float* out = partial + slice * kDim;
#pragma unroll
    for (int i = 0; i < kQueryCopies; ++i) {
        const int index = thread + i * kThreads;
        if (index < kTensorRows * kQuads) {
            const int row = index / kQuads;
            const int quad = index - row * kQuads;
            if (row < live) {
                store4(out + (long long)row * splits * kDim + quad * 4,
                       load4(out_tile + row * kTensorQueryStride + quad * 4));
            }
        }
    }

    if (kTimed && thread == 0) {
        long long* record = timing + 4 * (split * gridDim.x + head);
        record[0] = staged - entered;
        record[1] = scored - staged;
        record[2] = weighted - scored;
        record[3] = clock64() - entered;
    }
}

// Rescale one (head, row)'s slices onto their common maximum and write the
// transposed, rounded result.
//
// A warp owns one (head, row) and `kDim / kPieces` of its output dimensions, so
// both the partial read and the bf16 store stay contiguous. kPieces exists
// because the work is only heads * rows = 816 warps, which is a tenth of what
// an H100 wants resident: splitting the dimension axis buys parallelism at the
// price of narrower loads, and which way that lands is a measurement.
template <int kPieces>
__global__ void split_combine_kernel(
    const float* __restrict__ partial, const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum, __nv_bfloat16* __restrict__ target,
    int rows, int heads, int splits) {
    constexpr int kLanePiece = kDim / (kPieces * kWarpSize);   // floats per lane
    const int lane = threadIdx.x;
    const int slot = blockIdx.y * blockDim.y + threadIdx.y;
    const int head = slot / kPieces;
    if (head >= heads) {
        return;
    }
    const int row = blockIdx.x;
    const int offset = (slot - head * kPieces) * (kDim / kPieces) + lane * kLanePiece;

    // Two passes rather than one running rescale: the maxima are 4 bytes per
    // slice and the accumulators 512, so taking the maximum first leaves the
    // wide loads independent of each other. A running rescale would make each
    // iteration wait on the previous one's maximum, and this grid has nothing
    // else resident to cover that latency.
    //
    // The slices of one row are adjacent, so this row's maxima are one sector
    // and its accumulators one contiguous run of splits * 512 bytes.
    const long long base = ((long long)head * rows + row) * splits;
    float maximum = kMaskedLogit;
#pragma unroll 4
    for (int split = 0; split < splits; ++split) {
        maximum = fmaxf(maximum, partial_max[base + split]);
    }
    float total = 0.0f;
    float accumulator[kLanePiece];
#pragma unroll
    for (int e = 0; e < kLanePiece; ++e) {
        accumulator[e] = 0.0f;
    }
    // Four slices per unrolled step, so four wide loads are outstanding
    // together.
#pragma unroll 4
    for (int split = 0; split < splits; ++split) {
        const long long index = base + split;
        // A fully masked row reports the sentinel from every slice, so every
        // weight is exp(0) = 1 and the denominators sum to the full key count
        // -- the uniform distribution the finite sentinel exists to give.
        const float weight = __expf(partial_max[index] - maximum);
        const float* slice = partial + index * kDim + offset;
#pragma unroll
        for (int e = 0; e < kLanePiece; ++e) {
            accumulator[e] = fmaf(slice[e], weight, accumulator[e]);
        }
        total = fmaf(partial_sum[index], weight, total);
    }

    const float inverse = 1.0f / total;
    __nv_bfloat162* out = reinterpret_cast<__nv_bfloat162*>(
        target + (long long)row * heads * kDim + head * kDim + offset);
#pragma unroll
    for (int e = 0; e < kLanePiece; e += 2) {
        out[e / 2] = __floats2bfloat162_rn(accumulator[e] * inverse,
                                           accumulator[e + 1] * inverse);
    }
}

template <int kKeysPerWarp, int kRowGroups, int kRowTile, int kRowsPerWarp, int kUnroll = 2,
          bool kTimed = false>
int launch_split(const void* query, const void* keys, const void* values, const void* mask,
                 void* partial, void* partial_max, void* partial_sum,
                 int rows, int heads, int group, int key_count,
                 long long head_stride, long long key_stride, float scale,
                 int splits, cudaStream_t stream, long long* timing = nullptr) {
    constexpr int kKeyGroups = kWarps / kRowGroups;
    constexpr int kKeyTile = kKeysPerWarp * kKeyGroups;
    constexpr int kShared = shared_bytes(kKeyTile, kRowTile, kKeyGroups);
    auto kernel = split_attention_kernel<kKeysPerWarp, kRowGroups, kRowTile, kRowsPerWarp,
                                         kUnroll, kTimed>;
    // Past 48 KB a CTA has to opt in explicitly. Done once per instantiation;
    // the call is idempotent, so a benign race between host threads is harmless.
    static bool opted_in = false;
    if (!opted_in) {
        const cudaError_t status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
        if (status != cudaSuccess) {
            return (int)status;
        }
        opted_in = true;
    }
    const int row_tiles = (rows + kRowTile - 1) / kRowTile;
    kernel<<<dim3(heads, splits, row_tiles), dim3(kWarpSize, kWarps), kShared, stream>>>(
        (const float*)query, (const float*)keys, (const float*)values, (const bool*)mask,
        (float*)partial, (float*)partial_max, (float*)partial_sum, timing,
        rows, heads, group, key_count, head_stride, key_stride, scale);
    return (int)cudaPeekAtLastError();
}

// The PV pass gives each of the eight warps a fixed run of rows: seven covers
// the Target's 51 in a 64-row tile, eight covers a full one, four covers a
// 32-row tile.
template <int kKeysPerWarp, int kRowGroups, int kRowTile, int kUnroll = 2>
int launch_tile(const void* query, const void* keys, const void* values, const void* mask,
                void* partial, void* partial_max, void* partial_sum,
                int rows, int heads, int group, int key_count,
                long long head_stride, long long key_stride, float scale,
                int splits, cudaStream_t stream) {
    if (kRowTile == 32) {
        return launch_split<kKeysPerWarp, kRowGroups, kRowTile, kRowTile / kWarps, kUnroll>(
            query, keys, values, mask, partial, partial_max, partial_sum, rows, heads, group,
            key_count, head_stride, key_stride, scale, splits, stream);
    }
    if (rows <= 7 * kWarps) {
        return launch_split<kKeysPerWarp, kRowGroups, kRowTile, 7, kUnroll>(
            query, keys, values, mask, partial, partial_max, partial_sum, rows, heads, group,
            key_count, head_stride, key_stride, scale, splits, stream);
    }
    return launch_split<kKeysPerWarp, kRowGroups, kRowTile, 8, kUnroll>(
        query, keys, values, mask, partial, partial_max, partial_sum, rows, heads, group,
        key_count, head_stride, key_stride, scale, splits, stream);
}

template <int kKeyTile, bool kTimed = false>
int launch_tensor(const void* query, const void* keys, const void* values, const void* mask,
                  void* partial, void* partial_max, void* partial_sum,
                  int rows, int heads, int group, int key_count,
                  long long head_stride, long long key_stride, float scale,
                  int splits, cudaStream_t stream, long long* timing = nullptr) {
    constexpr int kShared = tensor_shared_bytes(kKeyTile);
    auto kernel = split_attention_tensor_kernel<kKeyTile, kTimed>;
    static bool opted_in = false;
    if (!opted_in) {
        const cudaError_t status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
        if (status != cudaSuccess) {
            return (int)status;
        }
        opted_in = true;
    }
    const int row_tiles = (rows + kTensorRows - 1) / kTensorRows;
    kernel<<<dim3(heads, splits, row_tiles), dim3(kWarpSize, kWarps), kShared, stream>>>(
        (const float*)query, (const float*)keys, (const float*)values, (const bool*)mask,
        (float*)partial, (float*)partial_max, (float*)partial_sum, timing,
        rows, heads, group, key_count, head_stride, key_stride, scale);
    return (int)cudaPeekAtLastError();
}

#define SPLIT_CASE(tile, rowtile, groups, perwarp)                                          \
    if (key_tile == (tile) && row_tile == (rowtile) && row_groups == (groups)) {            \
        return launch_tile<(perwarp), (groups), (rowtile)>(                                 \
            query, keys, values, mask, partial, partial_max, partial_sum, rows, heads,      \
            group, key_count, head_stride, key_stride, scale, splits, handle);              \
    }

int dispatch(const void* query, const void* keys, const void* values, const void* mask,
             void* partial, void* partial_max, void* partial_sum,
             int rows, int heads, int group, int key_count,
             long long head_stride, long long key_stride, float scale,
             int splits, int key_tile, int row_tile, int row_groups, int unroll, int tensor,
             cudaStream_t handle) {
    // The TF32 path fixes the CTA at 64 query rows and four 16-row MMA blocks,
    // so row_tile, row_groups and unroll do not apply to it. The key tile must
    // divide into two groups of whole 8-wide n tiles and fit one 64-bit mask
    // word, which leaves 32, 48 and 64.
    if (tensor) {
        switch (key_tile) {
            case 32: return launch_tensor<32>(query, keys, values, mask, partial, partial_max,
                                              partial_sum, rows, heads, group, key_count,
                                              head_stride, key_stride, scale, splits, handle);
            case 48: return launch_tensor<48>(query, keys, values, mask, partial, partial_max,
                                              partial_sum, rows, heads, group, key_count,
                                              head_stride, key_stride, scale, splits, handle);
            case 64: return launch_tensor<64>(query, keys, values, mask, partial, partial_max,
                                              partial_sum, rows, heads, group, key_count,
                                              head_stride, key_stride, scale, splits, handle);
            default: return -6;
        }
    }
    // Instantiated grid shapes. kKeysPerWarp is key_tile / (8 / row_groups).
    // Deeper unrolling of the QK mainloop puts more shared-memory loads in
    // flight at the cost of registers; measured only for the shipped shape.
    if (key_tile == 40 && row_tile == 64 && row_groups == 1 && unroll == 4) {
        return launch_tile<5, 1, 64, 4>(query, keys, values, mask, partial, partial_max,
                                        partial_sum, rows, heads, group, key_count,
                                        head_stride, key_stride, scale, splits, handle);
    }
    if (unroll != 2) {
        return -4;
    }
    SPLIT_CASE(24, 64, 1, 3)
    SPLIT_CASE(32, 64, 1, 4)
    SPLIT_CASE(40, 64, 1, 5)
    SPLIT_CASE(48, 64, 1, 6)
    SPLIT_CASE(64, 64, 1, 8)
    SPLIT_CASE(32, 64, 2, 8)
    SPLIT_CASE(40, 64, 2, 10)
    SPLIT_CASE(32, 32, 1, 4)
    SPLIT_CASE(40, 32, 1, 5)
    SPLIT_CASE(48, 32, 1, 6)
    return -3;
}

#undef SPLIT_CASE

}  // namespace

extern "C" int split_attention_partials(
    const void* query, const void* keys, const void* values, const void* mask,
    void* partial, void* partial_max, void* partial_sum,
    int rows, int heads, int kv_heads, int key_count, int dim,
    long long head_stride, long long key_stride, float scale,
    int key_tile, int row_tile, int row_groups, int unroll, int tensor, void* stream) {
    if (dim != kDim || kv_heads <= 0 || heads % kv_heads != 0) {
        return -1;
    }
    // Every float4 load and every cp.async assumes 16-byte alignment of a cache
    // row's start.
    if (head_stride % 4 != 0 || key_stride % 4 != 0) {
        return -2;
    }
    return dispatch(query, keys, values, mask, partial, partial_max, partial_sum, rows, heads,
                    heads / kv_heads, key_count, head_stride, key_stride, scale,
                    (key_count + key_tile - 1) / key_tile, key_tile, row_tile, row_groups,
                    unroll, tensor, (cudaStream_t)stream);
}

// `pieces` is how many warps share one (head, row); 1, 2 and 4 are
// instantiated, and 128 / (pieces * 32) floats per lane must stay even so the
// bf16 store can pair them.
extern "C" int split_attention_combine(
    const void* partial, const void* partial_max, const void* partial_sum, void* target,
    int rows, int heads, int splits, int pieces, void* stream) {
    constexpr int kCombineWarps = 8;
    const int slots = heads * pieces;
    const dim3 grid(rows, (slots + kCombineWarps - 1) / kCombineWarps);
    const dim3 block(kWarpSize, kCombineWarps);
    cudaStream_t handle = (cudaStream_t)stream;
    switch (pieces) {
        case 1:
            split_combine_kernel<1><<<grid, block, 0, handle>>>(
                (const float*)partial, (const float*)partial_max, (const float*)partial_sum,
                (__nv_bfloat16*)target, rows, heads, splits);
            break;
        case 2:
            split_combine_kernel<2><<<grid, block, 0, handle>>>(
                (const float*)partial, (const float*)partial_max, (const float*)partial_sum,
                (__nv_bfloat16*)target, rows, heads, splits);
            break;
        default:
            return -5;
    }
    return (int)cudaPeekAtLastError();
}

// `key_tile` selects the split width, `row_tile` the query rows a CTA covers,
// `row_groups` how its eight warps divide them, `unroll` the QK mainloop's
// unroll factor and `pieces` how many warps the combine gives one output row;
// the combination must be one of the instantiated grid shapes. `tensor` selects
// the TF32 tensor-core mainloops, which the upstream reference itself uses, in
// place of the float32 ones; it honours only `key_tile` and `pieces`.
// `partial` must hold heads * rows * splits * dim floats and
// `partial_max`/`partial_sum` heads * rows * splits each, indexed with the
// slice innermost, where splits is ceil(key_count / key_tile). Returns a
// nonzero CUDA error code on failure.
extern "C" int split_attention_launch(
    const void* query, const void* keys, const void* values, const void* mask,
    void* partial, void* partial_max, void* partial_sum, void* target,
    int rows, int heads, int kv_heads, int key_count, int dim,
    long long head_stride, long long key_stride, float scale,
    int key_tile, int row_tile, int row_groups, int unroll, int pieces, int tensor,
    void* stream) {
    const int code = split_attention_partials(query, keys, values, mask, partial, partial_max,
                                              partial_sum, rows, heads, kv_heads, key_count,
                                              dim, head_stride, key_stride, scale, key_tile,
                                              row_tile, row_groups, unroll, tensor, stream);
    if (code != 0) {
        return code;
    }
    return split_attention_combine(partial, partial_max, partial_sum, target, rows, heads,
                                   (key_count + key_tile - 1) / key_tile, pieces, stream);
}

// The workspace the caller must allocate, in floats, for one `key_tile`.
extern "C" int split_attention_splits(int key_count, int key_tile) {
    return (key_count + key_tile - 1) / key_tile;
}

// The instrumented variant of one grid shape at key_tile 40. `timing` must hold
// 4 * splits * heads int64s and receives, per CTA of the first row tile, the
// staging, QK, PV and whole-CTA cycle counts -- which also price the effective
// SM clock against a wall-clock measurement of the same launch.
extern "C" int split_attention_timed(
    const void* query, const void* keys, const void* values, const void* mask,
    void* partial, void* partial_max, void* partial_sum, void* timing,
    int rows, int heads, int kv_heads, int key_count, int dim,
    long long head_stride, long long key_stride, float scale, int row_groups, int tensor,
    void* stream) {
    if (dim != kDim || kv_heads <= 0 || heads % kv_heads != 0 || rows > 7 * kWarps) {
        return -1;
    }
    const int group = heads / kv_heads;
    if (tensor) {
        return launch_tensor<48, true>(query, keys, values, mask, partial, partial_max,
                                       partial_sum, rows, heads, group, key_count, head_stride,
                                       key_stride, scale, (key_count + 47) / 48,
                                       (cudaStream_t)stream, (long long*)timing);
    }
    const int splits = (key_count + 39) / 40;
    if (row_groups == 2) {
        return launch_split<10, 2, 64, 7, 2, true>(query, keys, values, mask, partial, partial_max,
                                                partial_sum, rows, heads, group, key_count,
                                                head_stride, key_stride, scale, splits,
                                                (cudaStream_t)stream, (long long*)timing);
    }
    return launch_split<5, 1, 64, 7, 2, true>(query, keys, values, mask, partial, partial_max,
                                           partial_sum, rows, heads, group, key_count,
                                           head_stride, key_stride, scale, splits,
                                           (cudaStream_t)stream, (long long*)timing);
}

// A do-nothing launch, so a benchmark can subtract the cost of a graph node.
__global__ void split_noop_kernel() {}

extern "C" int split_attention_noop(void* stream) {
    split_noop_kernel<<<1, 32, 0, (cudaStream_t)stream>>>();
    return (int)cudaPeekAtLastError();
}
