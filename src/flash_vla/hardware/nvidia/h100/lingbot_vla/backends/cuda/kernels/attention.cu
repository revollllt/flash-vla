// One fused masked GQA attention for the LingBot action expert's fixed shape.
//
// The expert's attention is 16 query heads over 51 suffix rows and 315 cached
// keys of 128 float32 dimensions, and it runs 360 times per forward. Split
// across two cuBLAS calls and a softmax it costs three launches and sends the
// 1 MB score tensor to DRAM and back twice: measured 7.08 + 4.56 + 6.85 =
// 18.5 us per layer-step (job 614367), against a ~2.4 us streaming floor for
// the 1.5 MB the operation actually touches and a ~2.2 us float32 FMA roofline
// for its 66 M multiply-accumulates. The scores never need to leave the SM.
//
// One warp owns one (head, query row) and holds that row's 128 queries in four
// registers per lane. It streams the key and value cache once, keeping the
// softmax in flash form -- a running maximum, a running denominator and a
// value accumulator rescaled to them -- so each score is produced, consumed and
// discarded. Four keys are scored per iteration so their four warp reductions
// interleave; with one reduction per key the shuffle latency, not the
// arithmetic, would set the pace.
//
// Arithmetic stays float32 throughout, as the upstream eager path's does; the
// reduction order differs from cuBLAS's, and the masked-logit sentinel is
// upstream's finite one, so a fully masked row still produces its uniform
// distribution.
//
// Layouts are the ones the projection epilogue already writes: `query` is
// [heads, rows, dim] and the caches are [kv_heads, keys, dim], float32 and
// contiguous in `dim`. The output is written transposed and rounded, as
// [rows, heads * dim] bf16, which is what the output projection consumes.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kWarpSize = 32;
// 16 heads x ceil(51 / 4) row tiles = 208 CTAs of four warps, which covers the
// machine's 132 SMs. Each lane owns dim/32 strided dimensions, so every global
// read of a key or value row is one coalesced warp transaction.
constexpr int kRowsPerBlock = 4;
constexpr int kKeysPerStep = 4;
constexpr int kMaxDimPerLane = 8;          // dim <= 256
constexpr float kMaskedLogit = -2.3819763e38f;

__global__ void expert_attention_kernel(
    const float* __restrict__ query, const float* __restrict__ keys,
    const float* __restrict__ values, const bool* __restrict__ mask,
    __nv_bfloat16* __restrict__ target,
    int rows, int heads, int kv_heads, int key_count, int dim, float scale) {
    const int head = blockIdx.x;
    const int row = blockIdx.y * kRowsPerBlock + threadIdx.y;
    if (row >= rows) {
        return;
    }
    const int lane = threadIdx.x;
    const int per_lane = dim / kWarpSize;
    const int kv = head / (heads / kv_heads);
    const float* key_base = keys + (long long)kv * key_count * dim;
    const float* value_base = values + (long long)kv * key_count * dim;
    const bool* row_mask = mask + (long long)row * key_count;

    float q[kMaxDimPerLane];
    float accumulator[kMaxDimPerLane];
    const float* q_row = query + ((long long)head * rows + row) * dim;
    for (int i = 0; i < per_lane; ++i) {
        q[i] = q_row[lane + i * kWarpSize];
        accumulator[i] = 0.0f;
    }
    float maximum = kMaskedLogit;
    float total = 0.0f;

    for (int base = 0; base < key_count; base += kKeysPerStep) {
        float dots[kKeysPerStep] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int active = min(kKeysPerStep, key_count - base);
        for (int i = 0; i < per_lane; ++i) {
            const int index = lane + i * kWarpSize;
            for (int t = 0; t < active; ++t) {
                dots[t] = __fmaf_rn(q[i], key_base[(long long)(base + t) * dim + index],
                                    dots[t]);
            }
        }
        // Four independent reductions issued together, so the shuffle latency
        // of one overlaps the arithmetic of the next.
        for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
            for (int t = 0; t < kKeysPerStep; ++t) {
                dots[t] += __shfl_xor_sync(0xffffffffu, dots[t], offset);
            }
        }
        for (int t = 0; t < active; ++t) {
            const float logit = row_mask[base + t] ? dots[t] * scale : kMaskedLogit;
            const float next = fmaxf(maximum, logit);
            const float rescale = __expf(maximum - next);
            const float weight = __expf(logit - next);
            maximum = next;
            total = __fmaf_rn(total, rescale, weight);
            const float* v_row = value_base + (long long)(base + t) * dim;
            for (int i = 0; i < per_lane; ++i) {
                accumulator[i] = __fmaf_rn(accumulator[i], rescale,
                                           weight * v_row[lane + i * kWarpSize]);
            }
        }
    }

    const float inverse = 1.0f / total;
    __nv_bfloat16* out = target + (long long)row * heads * dim + head * dim;
    for (int i = 0; i < per_lane; ++i) {
        out[lane + i * kWarpSize] = __float2bfloat16(accumulator[i] * inverse);
    }
}

}  // namespace

extern "C" int expert_attention_launch(
    const void* query, const void* keys, const void* values, const void* mask,
    void* target, int rows, int heads, int kv_heads, int key_count, int dim,
    float scale, void* stream) {
    if (dim % kWarpSize != 0 || dim / kWarpSize > kMaxDimPerLane) {
        return 1;
    }
    const dim3 grid(heads, (rows + kRowsPerBlock - 1) / kRowsPerBlock);
    const dim3 block(kWarpSize, kRowsPerBlock);
    expert_attention_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)query, (const float*)keys, (const float*)values,
        (const bool*)mask, (__nv_bfloat16*)target,
        rows, heads, kv_heads, key_count, dim, scale);
    return (int)cudaPeekAtLastError();
}
