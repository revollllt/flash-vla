// Hand-written pointwise stages of the LingBot action expert's attention block.
//
// Three launches replace twenty-four upstream issues per layer-step, and the
// expert pays every one of them 360 times per forward (36 layers x 10 denoise
// steps) at 51 rows, which is far under one wave of the machine:
//
//   expert_rope_launch      the float32 widening of the packed bf16 q/k/v GEMM,
//                           the two seven-kernel RoPE rotations, and the two
//                           writes of the 51 suffix rows into the resident
//                           key/value cache -- seventeen issues.
//   expert_softmax_launch   the scale, the mask select and the softmax over the
//                           315 keys, in place -- three issues and two extra
//                           round trips of the 1 MB score tensor.
//   expert_epilogue_launch  the group-major to token-major transpose of the
//                           attention output and its bf16 rounding for the
//                           output projection -- two issues.
//
// Arithmetic follows the upstream torch expressions instruction for
// instruction: the rotation issues separate multiplies and a round-to-nearest
// add or subtract (never an FMA), and the softmax keeps upstream's finite
// masked sentinel rather than skipping masked keys, so a fully masked row would
// still produce upstream's uniform distribution.
//
// Output addressing is (head_stride, token_stride) per tensor so one kernel
// serves both the token-major layout the reference path uses and the
// group-major layout the batched attention matmul wants.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kRopeThreads = 256;
constexpr int kSoftmaxThreads = 128;
constexpr int kEpilogueThreads = 256;
constexpr int kNormThreads = 256;
// Upstream's masked-logit sentinel: finite, so exp(sentinel - max) underflows
// to exactly zero unless every key in the row is masked.
constexpr float kMaskedLogit = -2.3819763e38f;

__global__ void expert_rope_kernel(
    const __nv_bfloat16* __restrict__ packed,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    float* __restrict__ query,
    float* __restrict__ key_slot,
    float* __restrict__ value_slot,
    int rows, int q_heads, int kv_heads, int head_dim,
    int q_head_stride, int q_token_stride,
    int kv_head_stride, int kv_token_stride,
    int rotate_total, int value_total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = head_dim >> 1;
    const int q_width = q_heads * head_dim;
    const int kv_width = kv_heads * head_dim;
    const int packed_width = q_width + 2 * kv_width;

    if (index < rotate_total) {
        const int per_row = (q_heads + kv_heads) * half;
        const int row = index / per_row;
        const int rest = index - row * per_row;
        const int head = rest / half;
        const int pair = rest - head * half;

        const __nv_bfloat16* source;
        float* target;
        if (head < q_heads) {
            source = packed + (long long)row * packed_width + head * head_dim;
            target = query + (long long)head * q_head_stride + (long long)row * q_token_stride;
        } else {
            const int kv = head - q_heads;
            source = packed + (long long)row * packed_width + q_width + kv * head_dim;
            target = key_slot + (long long)kv * kv_head_stride + (long long)row * kv_token_stride;
        }
        const float first = __bfloat162float(source[pair]);
        const float second = __bfloat162float(source[pair + half]);
        const float cosine = cos_table[row * half + pair];
        const float sine = sin_table[row * half + pair];
        target[pair] = __fsub_rn(__fmul_rn(first, cosine), __fmul_rn(second, sine));
        target[pair + half] = __fadd_rn(__fmul_rn(second, cosine), __fmul_rn(first, sine));
        return;
    }

    const int value_index = index - rotate_total;
    if (value_index < value_total) {
        const int row = value_index / kv_width;
        const int rest = value_index - row * kv_width;
        const int kv = rest / head_dim;
        const int lane = rest - kv * head_dim;
        value_slot[(long long)kv * kv_head_stride + (long long)row * kv_token_stride + lane] =
            __bfloat162float(packed[(long long)row * packed_width + q_width + kv_width + rest]);
    }
}

// One CTA per (head, query row). The row is 315 keys on this Target, so a
// single 128-thread pass over it keeps the whole reduction in registers and
// shared memory and never re-reads the score tensor from L2.
__global__ void expert_softmax_kernel(
    float* __restrict__ scores, const bool* __restrict__ mask,
    int keys, int q_rows, float scale) {
    __shared__ float reduction[kSoftmaxThreads];
    const int row = blockIdx.x;
    const int query_row = row % q_rows;
    float* line = scores + (long long)row * keys;
    const bool* line_mask = mask + (long long)query_row * keys;

    float best = kMaskedLogit;
    for (int i = threadIdx.x; i < keys; i += blockDim.x) {
        const float value = line_mask[i] ? line[i] * scale : kMaskedLogit;
        line[i] = value;
        best = fmaxf(best, value);
    }
    reduction[threadIdx.x] = best;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x],
                                           reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float maximum = reduction[0];
    __syncthreads();

    float total = 0.0f;
    for (int i = threadIdx.x; i < keys; i += blockDim.x) {
        const float value = expf(line[i] - maximum);
        line[i] = value;
        total += value;
    }
    reduction[threadIdx.x] = total;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float total_sum = reduction[0];
    for (int i = threadIdx.x; i < keys; i += blockDim.x) {
        line[i] /= total_sum;
    }
}

// [kv_heads, group, rows, head_dim] float32 -> [rows, kv_heads * group * head_dim] bf16.
__global__ void expert_epilogue_kernel(
    const float* __restrict__ source, __nv_bfloat16* __restrict__ target,
    int rows, int heads, int head_dim, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int lane = index % head_dim;
    const int rest = index / head_dim;
    const int row = rest % rows;
    const int head = rest / rows;
    target[(long long)row * heads * head_dim + head * head_dim + lane] =
        __float2bfloat16(source[index]);
}

// Upstream's `Qwen2RMSNorm`: normalize in float32, round to bf16, then scale by
// the bf16 weight -- two roundings, in that order. Eager torch spends five to
// six launches on it (a widening copy, a mean reduction over 48 CTAs, the
// reciprocal square root, the scale and the weight product) and re-reads the
// activation from DRAM at each one.
__global__ void rms_norm_kernel(
    const __nv_bfloat16* __restrict__ source, const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ target, int width, float epsilon) {
    __shared__ float reduction[kNormThreads];
    const long long base = (long long)blockIdx.x * width;

    float total = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float value = __bfloat162float(source[base + i]);
        total = __fadd_rn(total, __fmul_rn(value, value));
    }
    reduction[threadIdx.x] = total;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(reduction[0] / (float)width + epsilon);
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float normalized = __bfloat162float(
            __float2bfloat16(__bfloat162float(source[base + i]) * scale));
        target[base + i] = __float2bfloat16(normalized * __bfloat162float(weight[i]));
    }
}

// Upstream's `AdaRMSNorm` preceded by the residual add that always feeds it.
// The two are separate launches in torch, and the sum is needed again as the
// next residual, so this writes both: `total` for the residual chain and
// `target` for the projection that consumes the normalized value.
//
// Rounding order follows torch's: bf16 elementwise arithmetic is evaluated in
// float32 and rounded once per operation, and the FiLM modulation stays in
// float32 until the single store.
__global__ void ada_rms_add_kernel(
    const __nv_bfloat16* __restrict__ source, const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight, const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    __nv_bfloat16* __restrict__ total, __nv_bfloat16* __restrict__ target,
    int width, float epsilon) {
    __shared__ float reduction[kNormThreads];
    const long long base = (long long)blockIdx.x * width;

    float squares = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const __nv_bfloat16 sum = __float2bfloat16(
            __bfloat162float(source[base + i]) + __bfloat162float(residual[base + i]));
        total[base + i] = sum;
        const float value = __bfloat162float(sum);
        squares = __fadd_rn(squares, __fmul_rn(value, value));
    }
    reduction[threadIdx.x] = squares;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(reduction[0] / (float)width + epsilon);
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float normalized =
            __bfloat162float(total[base + i]) * scale * __bfloat162float(weight[i]);
        target[base + i] = __float2bfloat16(
            (1.0f + __bfloat162float(gamma[i])) * normalized + __bfloat162float(beta[i]));
    }
}

// `silu(gate) * up` over one packed gated projection, `[rows, 2 * width]` in
// and `[rows, width]` out. torch spends two launches and one extra round trip
// of the hidden tile on it.
__global__ void silu_multiply_kernel(
    const __nv_bfloat16* __restrict__ source, __nv_bfloat16* __restrict__ target,
    int width, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int row = index / width;
    const int lane = index - row * width;
    const long long base = (long long)row * 2 * width;
    const float gate = __bfloat162float(source[base + lane]);
    const float activated = __bfloat162float(
        __float2bfloat16(gate / (1.0f + expf(-gate))));
    target[index] = __float2bfloat16(
        activated * __bfloat162float(source[base + width + lane]));
}

// `Qwen2RMSNorm` preceded by the residual add that feeds it, for the towers
// whose normalization carries no FiLM modulation. Rounding order is upstream's:
// the normalized value is rounded to bf16 before the weight product, which is
// why this cannot be expressed as `ada_rms_add` with a zero gamma and beta.
__global__ void rms_norm_add_kernel(
    const __nv_bfloat16* __restrict__ source, const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ total, __nv_bfloat16* __restrict__ target,
    int width, float epsilon) {
    __shared__ float reduction[kNormThreads];
    const long long base = (long long)blockIdx.x * width;

    float squares = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const __nv_bfloat16 sum = __float2bfloat16(
            __bfloat162float(source[base + i]) + __bfloat162float(residual[base + i]));
        total[base + i] = sum;
        const float value = __bfloat162float(sum);
        squares = __fadd_rn(squares, __fmul_rn(value, value));
    }
    reduction[threadIdx.x] = squares;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(reduction[0] / (float)width + epsilon);
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float normalized = __bfloat162float(
            __float2bfloat16(__bfloat162float(total[base + i]) * scale));
        target[base + i] = __float2bfloat16(normalized * __bfloat162float(weight[i]));
    }
}

}  // namespace

extern "C" int rms_norm_add_launch(
    const void* source, const void* residual, const void* weight, void* total,
    void* target, int rows, int width, float epsilon, void* stream) {
    rms_norm_add_kernel<<<rows, kNormThreads, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)source, (const __nv_bfloat16*)residual,
        (const __nv_bfloat16*)weight, (__nv_bfloat16*)total, (__nv_bfloat16*)target,
        width, epsilon);
    return (int)cudaPeekAtLastError();
}

extern "C" int ada_rms_add_launch(
    const void* source, const void* residual, const void* weight, const void* gamma,
    const void* beta, void* total, void* target, int rows, int width, float epsilon,
    void* stream) {
    ada_rms_add_kernel<<<rows, kNormThreads, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)source, (const __nv_bfloat16*)residual,
        (const __nv_bfloat16*)weight, (const __nv_bfloat16*)gamma,
        (const __nv_bfloat16*)beta, (__nv_bfloat16*)total, (__nv_bfloat16*)target,
        width, epsilon);
    return (int)cudaPeekAtLastError();
}

extern "C" int silu_multiply_launch(
    const void* source, void* target, int rows, int width, void* stream) {
    const int total = rows * width;
    const int blocks = (total + kEpilogueThreads - 1) / kEpilogueThreads;
    silu_multiply_kernel<<<blocks, kEpilogueThreads, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)source, (__nv_bfloat16*)target, width, total);
    return (int)cudaPeekAtLastError();
}

extern "C" int rms_norm_launch(
    const void* source, const void* weight, void* target,
    int rows, int width, float epsilon, void* stream) {
    rms_norm_kernel<<<rows, kNormThreads, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)source, (const __nv_bfloat16*)weight,
        (__nv_bfloat16*)target, width, epsilon);
    return (int)cudaPeekAtLastError();
}

extern "C" int expert_rope_launch(
    const void* packed, const void* cos_table, const void* sin_table,
    void* query, void* key_slot, void* value_slot,
    int rows, int q_heads, int kv_heads, int head_dim,
    int q_head_stride, int q_token_stride, int kv_head_stride, int kv_token_stride,
    void* stream) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        return 1;
    }
    const int rotate_total = rows * (q_heads + kv_heads) * (head_dim >> 1);
    const int value_total = rows * kv_heads * head_dim;
    const int blocks = (rotate_total + value_total + kRopeThreads - 1) / kRopeThreads;
    expert_rope_kernel<<<blocks, kRopeThreads, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)packed, (const float*)cos_table, (const float*)sin_table,
        (float*)query, (float*)key_slot, (float*)value_slot,
        rows, q_heads, kv_heads, head_dim,
        q_head_stride, q_token_stride, kv_head_stride, kv_token_stride,
        rotate_total, value_total);
    return (int)cudaPeekAtLastError();
}

extern "C" int expert_softmax_launch(
    void* scores, const void* mask, int heads, int q_rows, int keys, float scale,
    void* stream) {
    expert_softmax_kernel<<<heads * q_rows, kSoftmaxThreads, 0, (cudaStream_t)stream>>>(
        (float*)scores, (const bool*)mask, keys, q_rows, scale);
    return (int)cudaPeekAtLastError();
}

extern "C" int expert_epilogue_launch(
    const void* source, void* target, int rows, int heads, int head_dim, void* stream) {
    const int total = rows * heads * head_dim;
    const int blocks = (total + kEpilogueThreads - 1) / kEpilogueThreads;
    expert_epilogue_kernel<<<blocks, kEpilogueThreads, 0, (cudaStream_t)stream>>>(
        (const float*)source, (__nv_bfloat16*)target, rows, heads, head_dim, total);
    return (int)cudaPeekAtLastError();
}
