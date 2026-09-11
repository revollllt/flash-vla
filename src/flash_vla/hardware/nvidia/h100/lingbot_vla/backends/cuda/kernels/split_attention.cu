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
// The grid is too small to fill the machine more than once, so a CTA runs at
// eight or sixteen warps on an SM and has almost no other work to hide latency
// behind. Two things follow, and both were measured with the `kTimed`
// instrumentation below (job 614654, key_tile 40, 1.72 GHz):
//   - Staging cost 6650 of the CTA's 18771 cycles when each thread waited on
//     its own global load before storing it to shared memory. The tiles are
//     copied with `cp.async` instead, so every copy a thread owns is in flight
//     at once and the phase becomes bandwidth- rather than latency-bound.
//   - `row_slots` picks between one CTA per SM covering 64 query rows and two
//     covering 32 each. The second doubles the warps available to a scheduler
//     at the cost of reading the key and value slices twice; which wins is a
//     measurement, not a derivation, so both are instantiated.
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
             int splits, int key_tile, int row_tile, int row_groups, int unroll,
             cudaStream_t handle) {
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
    SPLIT_CASE(48, 64, 2, 12)
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
    int key_tile, int row_tile, int row_groups, int unroll, void* stream) {
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
                    unroll, (cudaStream_t)stream);
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
// the combination must be one of the instantiated grid shapes.
// `partial` must hold heads * rows * splits * dim floats and
// `partial_max`/`partial_sum` heads * rows * splits each, indexed with the
// slice innermost, where splits is ceil(key_count / key_tile). Returns a
// nonzero CUDA error code on failure.
extern "C" int split_attention_launch(
    const void* query, const void* keys, const void* values, const void* mask,
    void* partial, void* partial_max, void* partial_sum, void* target,
    int rows, int heads, int kv_heads, int key_count, int dim,
    long long head_stride, long long key_stride, float scale,
    int key_tile, int row_tile, int row_groups, int unroll, int pieces, void* stream) {
    const int code = split_attention_partials(query, keys, values, mask, partial, partial_max,
                                              partial_sum, rows, heads, kv_heads, key_count,
                                              dim, head_stride, key_stride, scale, key_tile,
                                              row_tile, row_groups, unroll, stream);
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
    long long head_stride, long long key_stride, float scale, int row_groups, void* stream) {
    if (dim != kDim || kv_heads <= 0 || heads % kv_heads != 0 || rows > 7 * kWarps) {
        return -1;
    }
    const int splits = (key_count + 39) / 40;
    const int group = heads / kv_heads;
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
