#include <algorithm>
// Does programmatic dependent launch pay on sm_120, at Pi0's chain shape?
//
// `griddepcontrol` is accepted by ptxas for sm_120, sm_120f and sm_120a, and
// the runtime carries cudaLaunchAttributeProgrammaticStreamSerialization, so
// the instruction is available. Whether it BUYS anything is a separate
// question, and it depends on how much of the dependent kernel's work is
// independent of its producer.
//
// Pi0's action expert is the favourable case by construction: each call site
// streams a weight of several MB that has nothing to do with the previous
// kernel's output, and touches a 51-row activation that does. So the weight
// read -- the dominant cost -- can sit ABOVE the wait and overlap the
// producer's tail. This models exactly that: a chain of dependent kernels,
// each reading a large producer-independent buffer and a small producer-
// dependent one.
//
// Three variants, same arithmetic:
//   plain   -- ordinary launches, one stream
//   pdl     -- wait placed immediately before the first read of producer data,
//              trigger swept over the kernel's phase boundaries
//   graph   -- the same chain captured and replayed, which is how it deploys
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

//: Producer-independent bytes per kernel. 5.2 MB is the expert's QKV weight.
constexpr int64_t kWeightElems = 5'242'880 / 4;
//: Producer-dependent bytes. 51 x 1024 bf16 is the expert's activation.
constexpr int64_t kActElems = 51 * 1024 / 2;

// 0 plain
// 1 wait derived (after the producer-independent read) + trigger last
// 2 wait derived + trigger right after the producer-independent read
// 3 wait at the very TOP, trigger last -- what a wrapper around somebody
//   else's kernel can do, since it cannot reach inside to place the wait
template <int kMode>
__global__ void chain_kernel(const float *__restrict__ w, const float *__restrict__ in,
                             float *__restrict__ out, int64_t wn, int64_t an) {
  if (kMode == 3) {
#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
  }
  // Producer-INDEPENDENT work. Above the wait, so PDL overlaps it with the
  // producer's tail. This is the weight stream.
  float acc = 0.f;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < wn;
       i += int64_t(gridDim.x) * blockDim.x)
    acc += w[i];

  if (kMode == 2) {
#if __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
  }

  // The wait is DERIVED, not swept: it sits immediately before the first read
  // of producer data. Later would be a race, not a slower kernel.
  if (kMode == 1 || kMode == 2) {
#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
  }

  // Producer-DEPENDENT work.
  float dep = 0.f;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < an;
       i += int64_t(gridDim.x) * blockDim.x)
    dep += in[i];

  if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = acc + dep;
  if (blockIdx.x < 64 && threadIdx.x < 64)
    out[blockIdx.x * 64 + threadIdx.x] = acc * 1e-9f + dep;

  if (kMode == 1 || kMode == 3) {
#if __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
  }
}

template <int kMode>
static cudaError_t launch(dim3 grid, dim3 block, cudaStream_t s, bool pdl,
                          const float *w, const float *in, float *out) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  if (pdl) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  return cudaLaunchKernelEx(&cfg, chain_kernel<kMode>, w, in, out,
                            kWeightElems, kActElems);
}

int main() {
  cudaDeviceProp p;
  CHECK(cudaGetDeviceProperties(&p, 0));
  const int chain = 64, reps = 50;

  float *w, *bufs[2];
  CHECK(cudaMalloc(&w, kWeightElems * sizeof(float)));
  CHECK(cudaMemset(w, 0, kWeightElems * sizeof(float)));
  for (int i = 0; i < 2; ++i) {
    CHECK(cudaMalloc(&bufs[i], kActElems * sizeof(float)));
    CHECK(cudaMemset(bufs[i], 0, kActElems * sizeof(float)));
  }
  const dim3 grid(680), block(256);
  cudaStream_t s;
  CHECK(cudaStreamCreate(&s));

  auto run = [&](int mode, bool pdl, bool graph, double *us_per_kernel) -> int {
    auto once = [&]() {
      for (int i = 0; i < chain; ++i) {
        const float *in = bufs[i & 1];
        float *out = bufs[(i + 1) & 1];
        cudaError_t e = (mode == 0) ? launch<0>(grid, block, s, pdl, w, in, out)
                      : (mode == 1) ? launch<1>(grid, block, s, pdl, w, in, out)
                      : (mode == 2) ? launch<2>(grid, block, s, pdl, w, in, out)
                                    : launch<3>(grid, block, s, pdl, w, in, out);
        if (e != cudaSuccess) { printf("launch: %s\n", cudaGetErrorString(e)); }
      }
    };
    cudaGraphExec_t exec = nullptr;
    if (graph) {
      cudaGraph_t g;
      CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
      once();
      CHECK(cudaStreamEndCapture(s, &g));
      CHECK(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
    }
    for (int i = 0; i < 5; ++i) { if (graph) CHECK(cudaGraphLaunch(exec, s)); else once(); }
    CHECK(cudaStreamSynchronize(s));
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    std::vector<double> samples;
    for (int r = 0; r < reps; ++r) {
      CHECK(cudaEventRecord(a, s));
      if (graph) CHECK(cudaGraphLaunch(exec, s)); else once();
      CHECK(cudaEventRecord(b, s));
      CHECK(cudaEventSynchronize(b));
      float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, a, b));
      samples.push_back(ms * 1000.0 / chain);
    }
    std::sort(samples.begin(), samples.end());
    *us_per_kernel = samples[samples.size() / 2];
    return 0;
  };

  printf("  %s, %d SMs, chain of %d dependent kernels\n", p.name,
         p.multiProcessorCount, chain);
  printf("  each reads %.2f MB producer-INdependent + %.3f MB producer-dependent\n",
         kWeightElems * 4 / 1e6, kActElems * 4 / 1e6);
  printf("  %-34s %10s %10s\n", "variant", "us/kernel", "vs plain");
  struct Row { const char *name; int mode; bool pdl; bool graph; };
  const Row rows[] = {
      {"stream, no PDL", 0, false, false},
      {"stream, PDL trigger at end", 1, true, false},
      {"stream, PDL trigger after weights", 2, true, false},
      {"graph, no PDL", 0, false, true},
      {"graph, PDL trigger at end", 1, true, true},
      {"graph, PDL trigger after weights", 2, true, true},
      {"graph, PDL wait at the very top", 3, true, true},
  };
  double base_stream = 0, base_graph = 0;
  for (const Row &r : rows) {
    double us = 0;
    if (run(r.mode, r.pdl, r.graph, &us)) return 1;
    if (r.mode == 0 && !r.graph) base_stream = us;
    if (r.mode == 0 && r.graph) base_graph = us;
    const double base = r.graph ? base_graph : base_stream;
    printf("  %-34s %9.2f %9.3fx\n", r.name, us, base / us);
  }
  return 0;
}
