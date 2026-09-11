// Frozen copy of ada_rms_add_kernel from the Target's cuda/kernels/pointwise.cu,
// so this branch can check the fused prologue against the kernel it replaces
// without reaching into a checkout another agent is editing.
//
// CAUTION: ada_rms_add_kernel below is the version that was in pointwise.cu
// when the AdaRMS prologue was measured (jobs 615500-615564); the Target has
// since replaced it with a bfloat162 form using a warp-shuffle block reduce,
// whose reduction order is different.  It is kept as-is so that measurement
// stays reproducible, and because the prologue it was the bar for is a recorded
// negative.  Anything NEW compared against the normalisation has to re-copy.
//
// silu_multiply_kernel is the vectorised form the Target carries now, not the
// scalar one an earlier profile saw -- it pairs columns as bfloat162 and its
// silu_of() rounds the activation to bf16 before the multiply.
//
// It is a copy, not an include: the point of the comparison is that the fused
// arithmetic matches THIS text -- the bf16 rounding of the residual sum before
// the square, the per-thread stride-256 fadd_rn accumulation, the 256-way
// shared-memory tree, the rsqrtf of (sum / width + epsilon), and the two
// separate multiplies before the fused multiply-add. If pointwise.cu changes,
// this file has to be re-copied and the comparison re-run.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kNormThreads = 256;

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

__device__ __forceinline__ float silu_of(float x) {
    return __bfloat162float(__float2bfloat16(x / (1.0f + expf(-x))));
}

__global__ void silu_multiply_kernel(
    const __nv_bfloat16* __restrict__ source, __nv_bfloat16* __restrict__ target,
    int width, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int pairs = width >> 1;
    const int row = index / pairs;
    const int lane = index - row * pairs;
    const __nv_bfloat162* gate_row =
        (const __nv_bfloat162*)(source + (long long)row * 2 * width);
    const __nv_bfloat162 gate = gate_row[lane];
    const __nv_bfloat162 up = gate_row[pairs + lane];
    ((__nv_bfloat162*)target)[(long long)row * pairs + lane] = __floats2bfloat162_rn(
        silu_of(__low2float(gate)) * __low2float(up),
        silu_of(__high2float(gate)) * __high2float(up));
}

}  // namespace

extern "C" int reference_ada_rms_add_launch(
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

extern "C" int reference_silu_multiply_launch(const void* source, void* target,
                                              int rows, int width, void* stream) {
    const int total = rows * (width >> 1);
    const int blocks = (total + 256 - 1) / 256;
    silu_multiply_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)source, (__nv_bfloat16*)target, width, total);
    return (int)cudaPeekAtLastError();
}
