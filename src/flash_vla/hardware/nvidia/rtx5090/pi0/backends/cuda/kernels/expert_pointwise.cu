// Hand-written pointwise stages for Pi0's action expert on sm_120.
//
// The torch route spells each of these as a chain of library kernels: RMSNorm
// is five (square, mean, add, rsqrt, multiply), the RoPE scatter is about ten
// (view, cast, repeat, four multiplies, a subtract, an add, three copies), and
// the gated activation is three. At 180 layer-steps that is thousands of
// launches, and the profile attributes 6.7 ms to `norm_qkv_rope` against a
// 1.27 ms ceiling and 6.3 ms to `norm_gated_ffn` against 2.65.
//
// None of these kernels does much arithmetic. They exist to collapse launch
// count and memory round-trips, which is what the measured table says the
// expert is actually paying: an in-graph launch costs 0.45 us
// [launch.lat.dev.ramp] and a cold read 3.35 + MB/1.524 us [ld.bw.dev.dram].
//
// Shapes are Pi0's action expert: 51 tokens, 1024 wide, 8 query heads over one
// 256-wide KV head, 4096 feed-forward. Every tensor here is small enough that
// the grid is one CTA per row; the rows-per-launch is the thing being reduced,
// not the work per row.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

//: RMSNorm epsilon, inside the mean. Matches the reference `tl_rms_factor`.
constexpr float kRmsEps = 1e-6f;
//: The tanh GELU approximation written as a sigmoid: the two are the same
//: function, and the sigmoid form is one fewer transcendental on this part.
constexpr float kGeluC0 = 1.5957691216057308f;
constexpr float kGeluC1 = 0.044715f;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  return v;
}

// Block-wide sum over at most 32 warps, through one shuffle round per level.
// A shared-memory tree was measured 1.3-2.2x slower than shuffles for this
// shape on the H100 line (run-02); the same structure is used here rather than
// re-deriving it, and the ratio has NOT been re-measured on this part.
__device__ __forceinline__ float block_sum(float v, float *shared) {
  const int32_t lane = threadIdx.x & 31;
  const int32_t warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  const int32_t warps = (blockDim.x + 31) >> 5;
  v = (threadIdx.x < warps) ? shared[threadIdx.x] : 0.f;
  if (warp == 0) v = warp_sum(v);
  return v;
}

__device__ __forceinline__ float gelu_tanh(float x) {
  return x * __frcp_rn(1.0f + __expf(-(kGeluC0 * x * (1.0f + kGeluC1 * x * x))));
}

// out[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps)
//
// One CTA per row; each thread reads a 128-bit chunk (8 bf16) so the row is
// covered in one pass and the accumulator is fp32 throughout, which is what
// the reference does and what keeps the two within a bf16 ulp.
__global__ void rms_norm_kernel(const __nv_bfloat16 *__restrict__ x,
                                __nv_bfloat16 *__restrict__ out,
                                int32_t rows, int32_t cols) {
  __shared__ float reduction[32];
  const int32_t row = blockIdx.x;
  if (row >= rows) return;

  const int32_t vec = cols >> 3;  // 128-bit chunks per row
  const int4 *in4 = reinterpret_cast<const int4 *>(x + int64_t(row) * cols);
  int4 *out4 = reinterpret_cast<int4 *>(out + int64_t(row) * cols);

  float acc = 0.f;
  for (int32_t i = threadIdx.x; i < vec; i += blockDim.x) {
    const int4 raw = in4[i];
    const __nv_bfloat16 *v = reinterpret_cast<const __nv_bfloat16 *>(&raw);
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      const float f = __bfloat162float(v[j]);
      acc += f * f;
    }
  }
  acc = block_sum(acc, reduction);
  __shared__ float scale;
  if (threadIdx.x == 0) scale = rsqrtf(acc / float(cols) + kRmsEps);
  __syncthreads();
  const float s = scale;

  for (int32_t i = threadIdx.x; i < vec; i += blockDim.x) {
    int4 raw = in4[i];
    __nv_bfloat16 *v = reinterpret_cast<__nv_bfloat16 *>(&raw);
#pragma unroll
    for (int32_t j = 0; j < 8; ++j)
      v[j] = __float2bfloat16(__bfloat162float(v[j]) * s);
    out4[i] = raw;
  }
}

// Split a packed (rows, q_dim + 2 * head_dim) projection into Q, K and V,
// rotating Q and K by an interleaved cos/sin table and copying V.
//
// The rotation pairs ADJACENT columns (2p, 2p+1), which is exactly the layout
// of a `__nv_bfloat162`, so one 32-bit load carries a whole rotation pair and
// the packing costs nothing. `rope` is (rows, head_dim) holding
// [cos0, sin0, cos1, sin1, ...]; column j reads rope[row, j % head_dim].
__global__ void rope_scatter_kernel(const __nv_bfloat16 *__restrict__ packed,
                                    const __nv_bfloat16 *__restrict__ rope,
                                    __nv_bfloat16 *__restrict__ q,
                                    __nv_bfloat16 *__restrict__ k,
                                    __nv_bfloat16 *__restrict__ v,
                                    int32_t rows, int32_t q_dim,
                                    int32_t head_dim) {
  const int32_t row = blockIdx.x;
  if (row >= rows) return;
  const int32_t n = q_dim + 2 * head_dim;
  const int32_t pairs = n >> 1;
  const int32_t half = head_dim >> 1;

  const __nv_bfloat162 *src =
      reinterpret_cast<const __nv_bfloat162 *>(packed + int64_t(row) * n);
  const __nv_bfloat162 *rot =
      reinterpret_cast<const __nv_bfloat162 *>(rope + int64_t(row) * head_dim);

  for (int32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
    const __nv_bfloat162 in = src[p];
    const int32_t j0 = p << 1;
    if (j0 < q_dim + head_dim) {
      // Q and K rotate; the table repeats every head_dim columns.
      const __nv_bfloat162 cs = rot[p % half];
      const float x0 = __bfloat162float(in.x), x1 = __bfloat162float(in.y);
      const float c = __bfloat162float(cs.x), s = __bfloat162float(cs.y);
      const __nv_bfloat162 o =
          __floats2bfloat162_rn(x0 * c - x1 * s, x1 * c + x0 * s);
      if (j0 < q_dim)
        reinterpret_cast<__nv_bfloat162 *>(q + int64_t(row) * q_dim)[p] = o;
      else
        reinterpret_cast<__nv_bfloat162 *>(k + int64_t(row) * head_dim)
            [p - (q_dim >> 1)] = o;
    } else {
      // V is copied, not rotated.
      reinterpret_cast<__nv_bfloat162 *>(v + int64_t(row) * head_dim)
          [p - ((q_dim + head_dim) >> 1)] = in;
    }
  }
}

// out = gelu_tanh(gate) * up, element-wise over (rows, cols).
//
// A flat grid rather than one CTA per row: at 51 x 4096 the row count alone
// would leave the machine 3x under-occupied, and there is no cross-column
// dependence to keep a row together for.
__global__ void gelu_mul_kernel(const __nv_bfloat16 *__restrict__ gate,
                                const __nv_bfloat16 *__restrict__ up,
                                __nv_bfloat16 *__restrict__ out,
                                int64_t elements) {
  const int64_t vec = elements >> 3;
  const int4 *g4 = reinterpret_cast<const int4 *>(gate);
  const int4 *u4 = reinterpret_cast<const int4 *>(up);
  int4 *o4 = reinterpret_cast<int4 *>(out);
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < vec;
       i += int64_t(gridDim.x) * blockDim.x) {
    int4 gr = g4[i];
    const int4 ur = u4[i];
    __nv_bfloat16 *gv = reinterpret_cast<__nv_bfloat16 *>(&gr);
    const __nv_bfloat16 *uv = reinterpret_cast<const __nv_bfloat16 *>(&ur);
#pragma unroll
    for (int32_t j = 0; j < 8; ++j)
      gv[j] = __float2bfloat16(gelu_tanh(__bfloat162float(gv[j]))
                               * __bfloat162float(uv[j]));
    o4[i] = gr;
  }
}

}  // namespace

extern "C" {

// Every entry takes raw device pointers and an explicit stream so the launch is
// safe inside a CUDA-graph capture. Returns a cudaError_t as int.
int rms_norm_launch(const void *x, void *out, int rows, int cols,
                    void *stream) {
  if ((cols & 7) != 0) return cudaErrorInvalidValue;  // needs 128-bit chunks
  const int32_t threads = (cols >= 2048) ? 512 : 256;
  rms_norm_kernel<<<rows, threads, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)x, (__nv_bfloat16 *)out, rows, cols);
  return (int)cudaGetLastError();
}

int rope_scatter_launch(const void *packed, const void *rope, void *q, void *k,
                        void *v, int rows, int q_dim, int head_dim,
                        void *stream) {
  if ((head_dim & 1) != 0 || (q_dim & 1) != 0) return cudaErrorInvalidValue;
  const int32_t n = q_dim + 2 * head_dim;
  const int32_t threads = (n >> 1) >= 512 ? 512 : 256;
  rope_scatter_kernel<<<rows, threads, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)packed, (const __nv_bfloat16 *)rope,
      (__nv_bfloat16 *)q, (__nv_bfloat16 *)k, (__nv_bfloat16 *)v, rows, q_dim,
      head_dim);
  return (int)cudaGetLastError();
}

int gelu_mul_launch(const void *gate, const void *up, void *out,
                    long long elements, void *stream) {
  if ((elements & 7) != 0) return cudaErrorInvalidValue;
  const int32_t threads = 256;
  // Two CTAs per SM at the measured cold-read knee [ld.ctas.dev.knee]; the
  // grid-stride loop covers whatever is left.
  const int64_t vec = elements >> 3;
  int32_t blocks = (int32_t)((vec + threads - 1) / threads);
  if (blocks > 340) blocks = 340;
  if (blocks < 1) blocks = 1;
  gelu_mul_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)gate, (const __nv_bfloat16 *)up,
      (__nv_bfloat16 *)out, elements);
  return (int)cudaGetLastError();
}

}  // extern "C"

// ---------------------------------------------------------------------------
// LayerNorm, for the vision tower and the projector.
//
// The expert and the backbone normalise with RMS; SigLIP uses a full LayerNorm
// with an affine. Torch spells it as about six launches (mean, centre, square,
// mean, rsqrt, scale-and-shift) where this is one.
namespace {

constexpr float kLayerNormEps = 1e-5f;

// out[r, :] = (x[r, :] - mean) * rsqrt(var + eps) * w + b, reduced in fp32.
//
// One CTA per row, 128-bit loads, and the mean and the mean of squares taken in
// the same pass so the row is read once rather than twice.
__global__ void layer_norm_kernel(const __nv_bfloat16 *__restrict__ x,
                                  const __nv_bfloat16 *__restrict__ w,
                                  const __nv_bfloat16 *__restrict__ b,
                                  __nv_bfloat16 *__restrict__ out,
                                  int32_t rows, int32_t cols) {
  __shared__ float reduction[32];
  __shared__ float shift, scale;
  const int32_t row = blockIdx.x;
  if (row >= rows) return;

  const int32_t vec = cols >> 3;
  const int4 *in4 = reinterpret_cast<const int4 *>(x + int64_t(row) * cols);
  int4 *out4 = reinterpret_cast<int4 *>(out + int64_t(row) * cols);

  float s = 0.f, sq = 0.f;
  for (int32_t i = threadIdx.x; i < vec; i += blockDim.x) {
    const int4 raw = in4[i];
    const __nv_bfloat16 *vv = reinterpret_cast<const __nv_bfloat16 *>(&raw);
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      const float f = __bfloat162float(vv[j]);
      s += f;
      sq += f * f;
    }
  }
  s = block_sum(s, reduction);
  __syncthreads();
  sq = block_sum(sq, reduction);
  if (threadIdx.x == 0) {
    const float mean = s / float(cols);
    shift = mean;
    scale = rsqrtf(fmaxf(sq / float(cols) - mean * mean, 0.f) + kLayerNormEps);
  }
  __syncthreads();
  const float mu = shift, inv = scale;

  for (int32_t i = threadIdx.x; i < vec; i += blockDim.x) {
    int4 raw = in4[i];
    __nv_bfloat16 *vv = reinterpret_cast<__nv_bfloat16 *>(&raw);
    const int32_t base = i << 3;
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      const float n = (__bfloat162float(vv[j]) - mu) * inv;
      vv[j] = __float2bfloat16(n * __bfloat162float(w[base + j])
                               + __bfloat162float(b[base + j]));
    }
    out4[i] = raw;
  }
}

}  // namespace

extern "C" int layer_norm_launch(const void *x, const void *w, const void *b,
                                 void *out, int rows, int cols, void *stream) {
  if ((cols & 7) != 0) return cudaErrorInvalidValue;
  const int32_t threads = (cols >= 2048) ? 512 : 256;
  layer_norm_kernel<<<rows, threads, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)w,
      (const __nv_bfloat16 *)b, (__nv_bfloat16 *)out, rows, cols);
  return (int)cudaGetLastError();
}

// Plain GELU in place, for the vision feed-forward expansion, whose activation
// has no gate to multiply against.
namespace {
__global__ void gelu_kernel(__nv_bfloat16 *__restrict__ x, int64_t elements) {
  const int64_t vec = elements >> 3;
  int4 *x4 = reinterpret_cast<int4 *>(x);
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < vec;
       i += int64_t(gridDim.x) * blockDim.x) {
    int4 raw = x4[i];
    __nv_bfloat16 *v = reinterpret_cast<__nv_bfloat16 *>(&raw);
#pragma unroll
    for (int32_t j = 0; j < 8; ++j)
      v[j] = __float2bfloat16(gelu_tanh(__bfloat162float(v[j])));
    x4[i] = raw;
  }
}
}  // namespace

extern "C" int gelu_launch(void *x, long long elements, void *stream) {
  if ((elements & 7) != 0) return cudaErrorInvalidValue;
  const int64_t vec = elements >> 3;
  int32_t blocks = (int32_t)((vec + 255) / 256);
  if (blocks > 340) blocks = 340;   // 2x SM count [ld.ctas.dev.knee]
  if (blocks < 1) blocks = 1;
  gelu_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>((__nv_bfloat16 *)x,
                                                        elements);
  return (int)cudaGetLastError();
}

// Gated activation over a PACKED projection: the gate and the up halves are
// the two halves of one row rather than two tensors.
//
// Packing them lets the feed-forward run one GEMM instead of two, which at the
// expert's shape is nearly free: 51 x 1024 x 8192 measured 10.31 us against
// 10.34 for 51 x 1024 x 4096, so the second half of the weights rides along.
// A skinny GEMM there costs about 8 us before it reads anything, and that fixed
// cost is what packing pays once instead of twice.
namespace {
__global__ void gelu_mul_packed_kernel(const __nv_bfloat16 *__restrict__ packed,
                                       __nv_bfloat16 *__restrict__ out,
                                       int32_t rows, int32_t half) {
  // Eight columns per thread, so every access is 128-bit. The scalar form this
  // replaced moved the same bytes at a quarter of the rate -- 61.72 us against
  // `gelu_mul`'s 14.52 for the same 75.5 MB -- purely because it issued one
  // 2-byte access per element. The gate and up halves are `half` apart within
  // a row, and `half` is a multiple of 8, so both loads stay 16-byte aligned.
  const int32_t vhalf = half >> 3;
  const int64_t total = int64_t(rows) * vhalf;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += int64_t(gridDim.x) * blockDim.x) {
    const int32_t r = int32_t(i / vhalf), c = int32_t(i - int64_t(r) * vhalf) * 8;
    const int64_t base = int64_t(r) * (int64_t(half) << 1) + c;
    int4 g = *(const int4 *)(packed + base);
    const int4 u = *(const int4 *)(packed + base + half);
    __nv_bfloat16 *gv = reinterpret_cast<__nv_bfloat16 *>(&g);
    const __nv_bfloat16 *uv = reinterpret_cast<const __nv_bfloat16 *>(&u);
#pragma unroll
    for (int32_t j = 0; j < 8; ++j)
      gv[j] = __float2bfloat16(gelu_tanh(__bfloat162float(gv[j]))
                               * __bfloat162float(uv[j]));
    *(int4 *)(out + int64_t(r) * half + c) = g;
  }
}
}  // namespace

extern "C" int gelu_mul_packed_launch(const void *packed, void *out, int rows,
                                      int half, void *stream) {
  if ((half & 7) != 0) return cudaErrorInvalidValue;  // needs 128-bit chunks
  const int32_t threads = 256;
  const int64_t total = int64_t(rows) * (half >> 3);
  int32_t blocks = (int32_t)((total + threads - 1) / threads);
  if (blocks > 340) blocks = 340;   // 2x SM count [ld.ctas.dev.knee]
  if (blocks < 1) blocks = 1;
  gelu_mul_packed_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)packed, (__nv_bfloat16 *)out, rows, half);
  return (int)cudaGetLastError();
}
