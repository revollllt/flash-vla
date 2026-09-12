// launch unit -- what a kernel costs before it moves a byte, and what a cold
// read costs per byte, on this device.
//
// This is the constant the whole repository is denominated in. sm90's
// [ld.bw.dev.dram] is `t_us = 1.85 + MB/2.77`, and every fusion decision in the
// project prices a removed launch against it. That number is an H100 fact; this
// probe measures the sm_120 one.
//
// Three questions, each with the hypothesis it kills:
//
//   L0  How repeatable is a measurement here at all?
//       Kills "this machine is quiet enough to compare two numbers". Reported
//       as the spread of the SAME point measured in separate sweeps, which is
//       the error bar every other number in this unit inherits. sm90 carries
//       ~6% and unpinnable clocks; nothing is assumed here.
//
//   L1  What does an empty launch cost, against grid size?
//       Kills "launch count is free once the work is big". Measured with an
//       empty kernel so nothing but the launch is in the number.
//
//   L2  What does a cold read cost, against bytes and against CTA count?
//       Kills "bytes / peak bandwidth is a floor". Fits `t = a + MB/b` over a
//       size sweep with L2 evicted before every launch, and sweeps CTA count at
//       fixed bytes to find the knee.
//
// Isolation: no tensor core, no shared memory, no barriers, no atomics. The
// read kernel does a strided vectorised load and a single guarded store that
// never fires, so the loads cannot be eliminated but nothing is written back.
// L2 is 96 MB here, so the flush buffer must exceed it or a "cold" read is warm.
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(x)                                                               \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      printf("CUDA error %s at line %d: %s\n", #x, __LINE__,                   \
             cudaGetErrorString(e_));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void empty_kernel() {}

// Read `n4` float4s with a grid-stride loop. The accumulator is stored only on
// an impossible condition, so ptxas cannot drop the loads and no store traffic
// enters the measurement.
__global__ void stream_read(const float4 *__restrict__ src, size_t n4,
                            float *__restrict__ sink) {
  float acc = 0.f;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
       i += stride) {
    float4 v = src[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 1.0e37f) sink[0] = acc;
}

struct Flusher {
  char *buf = nullptr;
  size_t bytes = 0;
  void init(size_t l2_bytes) {
    bytes = 2 * l2_bytes;  // twice L2, so a flush cannot leave the set resident
    CHECK(cudaMalloc(&buf, bytes));
  }
  void flush() { CHECK(cudaMemsetAsync(buf, 0, bytes)); }
};

static float time_kernel(void (*launch)(void *), void *arg, int reps,
                         Flusher *flusher) {
  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a));
  CHECK(cudaEventCreate(&b));
  std::vector<float> samples;
  samples.reserve(reps);
  for (int i = 0; i < reps; ++i) {
    if (flusher) flusher->flush();
    CHECK(cudaEventRecord(a));
    launch(arg);
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    samples.push_back(ms * 1000.f);  // us
  }
  CHECK(cudaEventDestroy(a));
  CHECK(cudaEventDestroy(b));
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

struct EmptyArg { int grid, block; };
static void launch_empty(void *p) {
  EmptyArg *a = (EmptyArg *)p;
  empty_kernel<<<a->grid, a->block>>>();
}

struct ReadArg { const float4 *src; size_t n4; float *sink; int grid, block; };
static void launch_read(void *p) {
  ReadArg *a = (ReadArg *)p;
  stream_read<<<a->grid, a->block>>>(a->src, a->n4, a->sink);
}

int main(int argc, char **argv) {
  int reps = (argc > 1) ? atoi(argv[1]) : 50;
  int sweeps = (argc > 2) ? atoi(argv[2]) : 3;

  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("# launch unit -- %s, %d SMs, L2 %.0f MB\n", prop.name,
         prop.multiProcessorCount, prop.l2CacheSize / 1048576.0);
  printf("# reps=%d (median), sweeps=%d (for the spread), L2 flushed before "
         "every cold launch\n\n", reps, sweeps);
  const int SMS = prop.multiProcessorCount;

  Flusher flusher;
  flusher.init(prop.l2CacheSize);

  float *sink = nullptr;
  CHECK(cudaMalloc(&sink, sizeof(float)));

  // ---- L1: empty-kernel launch cost vs grid size -------------------------
  printf("## L1 empty launch (us), 256 threads/CTA\n");
  printf("%10s", "CTAs");
  for (int s = 0; s < sweeps; ++s) printf(" %9s%d", "sweep", s);
  printf("  %9s %7s\n", "median", "spread%");
  for (int grid : {1, 32, 64, 128, 170, 256, 340, 512, 1024}) {
    EmptyArg arg{grid, 256};
    launch_empty(&arg);
    CHECK(cudaDeviceSynchronize());  // warm
    std::vector<float> v;
    printf("%10d", grid);
    for (int s = 0; s < sweeps; ++s) {
      float t = time_kernel(launch_empty, &arg, reps, nullptr);
      v.push_back(t);
      printf("  %9.3f", t);
    }
    std::sort(v.begin(), v.end());
    float spread = 100.f * (v.back() - v.front()) / v[v.size() / 2];
    printf("  %9.3f %6.1f%%\n", v[v.size() / 2], spread);
  }

  // ---- L1b: launch-to-launch cost, event overhead removed ----------------
  // L1 brackets ONE launch with two events, so it measures launch + event
  // overhead and is an upper bound. Timing a batch and dividing removes the
  // per-launch event cost and leaves the back-to-back launch interval, which
  // is what a fusion decision actually trades against.
  printf("\n## L1b launch-to-launch interval (us/launch), batch of 200\n");
  printf("%10s %12s %12s %8s\n", "CTAs", "batched", "L1 single", "delta");
  for (int grid : {1, 170, 340, 1024}) {
    const int BATCH = 200;
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    empty_kernel<<<grid, 256>>>();
    CHECK(cudaDeviceSynchronize());
    std::vector<float> v;
    for (int s = 0; s < sweeps; ++s) {
      CHECK(cudaEventRecord(a));
      for (int i = 0; i < BATCH; ++i) empty_kernel<<<grid, 256>>>();
      CHECK(cudaEventRecord(b));
      CHECK(cudaEventSynchronize(b));
      float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, a, b));
      v.push_back(ms * 1000.f / BATCH);
    }
    std::sort(v.begin(), v.end());
    EmptyArg arg{grid, 256};
    float single = time_kernel(launch_empty, &arg, reps, nullptr);
    printf("%10d %12.3f %12.3f %7.2fx\n", grid, v[v.size()/2], single,
           single / v[v.size()/2]);
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
  }

  // ---- L1c: launch cost INSIDE a CUDA graph ------------------------------
  // The deployed path in this repository captures its whole forward into a CUDA
  // graph and replays it, so the cost that a fusion decision actually trades
  // against is a graph node's, not a stream launch's. sm90's note says the ramp
  // is "not removed by graph capture"; that is a claim about that machine and
  // is measured here rather than assumed.
  printf("\n## L1c launch inside a CUDA graph (us/launch), 64 nodes\n");
  printf("%10s %12s %12s %8s\n", "CTAs", "in graph", "in stream", "graph/stream");
  for (int grid : {1, 170, 340, 1024}) {
    const int NODES = 64;
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < NODES; ++i) empty_kernel<<<grid, 256, 0, stream>>>();
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    std::vector<float> v;
    for (int s = 0; s < sweeps; ++s) {
      CHECK(cudaEventRecord(a, stream));
      CHECK(cudaGraphLaunch(exec, stream));
      CHECK(cudaEventRecord(b, stream));
      CHECK(cudaEventSynchronize(b));
      float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, a, b));
      v.push_back(ms * 1000.f / NODES);
    }
    std::sort(v.begin(), v.end());

    const int BATCH = 200;
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < BATCH; ++i) empty_kernel<<<grid, 256>>>();
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, a, b));
    const float streamed = ms * 1000.f / BATCH;

    printf("%10d %12.3f %12.3f %7.2fx\n", grid, v[v.size()/2], streamed,
           v[v.size()/2] / streamed);
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
  }

  // ---- L2a: cold read cost vs bytes -> fits t = a + MB/b ------------------
  printf("\n## L2a cold read, 4x SM CTAs x 256 threads\n");
  printf("%10s %10s", "MB", "us");
  for (int s = 1; s < sweeps; ++s) printf(" %9s%d", "sweep", s);
  printf("  %9s %7s\n", "GB/s", "spread%");
  const int READ_GRID = 4 * SMS;
  std::vector<std::pair<double, double>> fit;  // (MB, us)
  for (double mb : {0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0}) {
    size_t bytes = size_t(mb * 1048576.0);
    size_t n4 = bytes / sizeof(float4);
    float4 *src = nullptr;
    if (cudaMalloc(&src, bytes) != cudaSuccess) { printf("%10.1f  (alloc failed)\n", mb); continue; }
    CHECK(cudaMemset(src, 1, bytes));
    ReadArg arg{src, n4, sink, READ_GRID, 256};
    launch_read(&arg);
    CHECK(cudaDeviceSynchronize());
    std::vector<float> v;
    printf("%10.1f", mb);
    for (int s = 0; s < sweeps; ++s) {
      float t = time_kernel(launch_read, &arg, reps, &flusher);
      v.push_back(t);
      printf(" %10.3f", t);
    }
    std::sort(v.begin(), v.end());
    float med = v[v.size() / 2];
    float spread = 100.f * (v.back() - v.front()) / med;
    printf("  %9.1f %6.1f%%\n", mb * 1048576.0 / (med * 1e-6) / 1e9, spread);
    fit.push_back({mb, med});
    CHECK(cudaFree(src));
  }

  // ---- L2c: the same cold read, timed as a CUDA graph replay -------------
  // L2a launches from a stream, so its fixed cost carries the stream-launch
  // overhead L1c measures at ~2.05 us. The deployed path replays a captured
  // graph, where a launch costs a quarter of that, so the floor model needs
  // the in-graph fixed cost. The L2 flush stays OUTSIDE the timed region.
  printf("\n## L2c cold read as a 1-node graph replay (the deployed shape)\n");
  printf("%10s %12s %12s %8s\n", "MB", "in graph", "in stream", "delta us");
  std::vector<std::pair<double, double>> gfit;
  for (double mb : {0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0}) {
    size_t bytes = size_t(mb * 1048576.0);
    size_t n4 = bytes / sizeof(float4);
    float4 *src = nullptr;
    if (cudaMalloc(&src, bytes) != cudaSuccess) continue;
    CHECK(cudaMemset(src, 1, bytes));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaGraph_t graph; cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    stream_read<<<READ_GRID, 256, 0, stream>>>(src, n4, sink);
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    std::vector<float> v;
    for (int i = 0; i < reps; ++i) {
      flusher.flush();
      CHECK(cudaStreamSynchronize(0));
      CHECK(cudaEventRecord(a, stream));
      CHECK(cudaGraphLaunch(exec, stream));
      CHECK(cudaEventRecord(b, stream));
      CHECK(cudaEventSynchronize(b));
      float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, a, b));
      v.push_back(ms * 1000.f);
    }
    std::sort(v.begin(), v.end());
    float g = v[v.size() / 2];
    double st = 0; for (auto &q : fit) if (q.first == mb) st = q.second;
    printf("%10.1f %12.3f %12.3f %8.3f\n", mb, g, st, st - g);
    gfit.push_back({mb, g});
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    CHECK(cudaGraphExecDestroy(exec)); CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream)); CHECK(cudaFree(src));
  }

  // Least squares on t = a + MB/b, over the whole sweep.
  auto do_fit = [&](double min_mb, const char *label) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
    for (auto &p : fit) {
      if (p.first < min_mb) continue;
      sx += p.first; sy += p.second; sxx += p.first * p.first;
      sxy += p.first * p.second; n += 1;
    }
    if (n < 2) return;
    double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);   // us per MB
    double intercept = (sy - slope * sx) / n;
    // MB/us -> TB/s: one MB per us is 1.048576e12 B/s.
    double tbs = (1.0 / slope) * 1.048576;
    printf("  %-28s t_us = %.3f + MB/%.3f   (%.3f TB/s marginal)\n",
           label, intercept, 1.0 / slope, tbs);
  };
  if (fit.size() >= 2) {
    printf("\n  Small sizes are fixed-cost dominated, so the whole-range fit\n"
           "  understates the marginal rate. Both are shown.\n");
    do_fit(0.0, "stream, all points:");
    do_fit(16.0, "stream, >= 16 MB:");
    fit.swap(gfit);
    do_fit(16.0, "GRAPH, >= 16 MB:   <-- use");
    fit.swap(gfit);
    printf("  %-28s t_us = 1.850 + MB/2.770   (2.905 TB/s marginal)\n",
           "sm90 [ld.bw.dev.dram]:");
    printf("  datasheet peak here is 1.792 TB/s (spec.py)\n");
  }

  // ---- L2b: cold read vs CTA count at fixed bytes -------------------------
  printf("\n## L2b cold read of 16 MB vs CTA count (256 threads/CTA)\n");
  printf("%10s %10s %10s %8s\n", "CTAs", "us", "GB/s", "vs best");
  {
    size_t bytes = 16u * 1048576u;
    size_t n4 = bytes / sizeof(float4);
    float4 *src = nullptr;
    CHECK(cudaMalloc(&src, bytes));
    CHECK(cudaMemset(src, 1, bytes));
    std::vector<std::pair<int, float>> rows;
    for (int grid : {SMS / 4, SMS / 2, SMS, 2 * SMS, 4 * SMS, 8 * SMS, 16 * SMS}) {
      ReadArg arg{src, n4, sink, grid, 256};
      launch_read(&arg);
      CHECK(cudaDeviceSynchronize());
      float t = time_kernel(launch_read, &arg, reps, &flusher);
      rows.push_back({grid, t});
    }
    float best = rows[0].second;
    for (auto &r : rows) best = std::min(best, r.second);
    for (auto &r : rows)
      printf("%10d %10.3f %10.1f %7.2fx\n", r.first, r.second,
             16.0 * 1048576.0 / (r.second * 1e-6) / 1e9, r.second / best);
    CHECK(cudaFree(src));
  }

  printf("\n# L0: the spread%% columns above ARE the noise floor. Every number\n"
         "# in this unit inherits the worst of them as its error bar.\n");
  return 0;
}
