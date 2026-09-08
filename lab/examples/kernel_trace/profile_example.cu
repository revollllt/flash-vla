// Educational CUDA trace example; not a Flash-VLA integration or benchmark.
// Build from the repository root with -Isrc and -arch=sm_90.
#include "flash_vla/hardware/nvidia/debug/kernel_trace.cuh"
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t err, const char* operation) {
  if (err != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(err));
}
#define CUDA_CHECK(x) check((x), #x)

template<class T> struct DeviceBuffer {
  T* p = nullptr;
  std::size_t n;
  explicit DeviceBuffer(std::size_t count) : n(count) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
  }
  ~DeviceBuffer() { if (p) cudaFree(p); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct Stream {
  cudaStream_t s;
  Stream() { CUDA_CHECK(cudaStreamCreate(&s)); }
  ~Stream() { cudaStreamDestroy(s); }
};

// Two independent, warp-uniform roles inside a CTA. Role intervals can overlap
// in wall time; this does NOT prove cycle-by-cycle simultaneous engine execution.
// This uses synchronous WMMA, NOT a Hopper TMA/WGMMA async pipeline.
template<bool Trace, bool Focused = false>
__global__ void two_role_kernel(const half* a, const half* b,
                                float* scalar_out, float* tensor_out,
                                flash_trace::RangeRecord* records, int iterations) {
  const unsigned warp = threadIdx.x / 32;
  const unsigned lane = threadIdx.x % 32;
  flash_trace::Stamp begin{};
  if constexpr (Trace) {
    if (lane == 0 && (!Focused || blockIdx.x == 0)) begin = flash_trace::read_stamp();
  }
  if (warp == 0) {
    float x = __half2float(a[lane]) + float(blockIdx.x % 8) * 0.01f;
    #pragma unroll 1
    for (int i = 0; i < iterations * 32; ++i)
      x = fmaf(x, 0.999f, 0.0007f);
    scalar_out[blockIdx.x * 32 + lane] = x;
  } else {  // all 32 lanes of warp 1 participate in each WMMA operation
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::load_matrix_sync(af, a, 16);
    wmma::load_matrix_sync(bf, b, 16);
    wmma::fill_fragment(cf, 0.0f);
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(tensor_out + blockIdx.x * 256, cf, 16, wmma::mem_row_major);
  }
  if constexpr (Trace) {
    if (lane == 0 && (!Focused || blockIdx.x == 0)) {
      const auto end = flash_trace::read_stamp();
      // Unique fixed slot for (CTA, logical warp); no global atomic counter.
      flash_trace::store_range(&records[blockIdx.x * 2 + warp], begin, end,
                              blockIdx.x, warp, warp + 1);
    }
  }
}

template<class T> std::vector<T> copy_to_host(const DeviceBuffer<T>& b) {
  std::vector<T> v(b.n);
  CUDA_CHECK(cudaMemcpy(v.data(), b.p, b.n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}

// 'profile' runs parity then captures ONE selected launch. Reset, allocation,
// synchronization and export are outside the kernel. No latency claim is made.
static void profile(const std::string& output, int blocks, int iterations) {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major < 7) throw std::runtime_error("WMMA demonstration needs SM70+");
  Stream stream;
  DeviceBuffer<half> a(256), b(256);
  DeviceBuffer<float> scalar(std::size_t(blocks) * 32), tensor(std::size_t(blocks) * 256);
  DeviceBuffer<flash_trace::RangeRecord> records(std::size_t(blocks) * 2);
  std::vector<half> ones(256, __float2half(1.0f));
  CUDA_CHECK(cudaMemcpy(a.p, ones.data(), a.n*sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.p, ones.data(), b.n*sizeof(half), cudaMemcpyHostToDevice));
  for (int i = 0; i < 3; ++i) {
    two_role_kernel<false><<<blocks, 64, 0, stream.s>>>(a.p, b.p, scalar.p, tensor.p, nullptr, iterations);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaStreamSynchronize(stream.s));
  const auto ref_s = copy_to_host(scalar);
  const auto ref_t = copy_to_host(tensor);
  // Warm the separately compiled traced function before the selected capture.
  two_role_kernel<true><<<blocks, 64, 0, stream.s>>>(a.p, b.p, scalar.p, tensor.p, records.p, iterations);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemsetAsync(records.p, 0, records.n*sizeof(flash_trace::RangeRecord), stream.s));
  two_role_kernel<true><<<blocks, 64, 0, stream.s>>>(a.p, b.p, scalar.p, tensor.p, records.p, iterations);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream.s));
  const auto out_s = copy_to_host(scalar);
  const auto out_t = copy_to_host(tensor);
  if (std::memcmp(ref_s.data(), out_s.data(), out_s.size()*sizeof(float)) ||
      std::memcmp(ref_t.data(), out_t.data(), out_t.size()*sizeof(float)))
    throw std::runtime_error("trace-on/off bitwise parity failed");
  for (float v : out_t)
    if (v != 16.0f * iterations) throw std::runtime_error("WMMA numerical check failed");
  for (float v : out_s)
    if (!std::isfinite(v)) throw std::runtime_error("scalar result is not finite");
  const auto ranges = copy_to_host(records);
  std::ofstream f(output);
  if (!f) throw std::runtime_error("cannot open output path: " + output);
  f << "{\"schema_version\":1,\"synthetic\":false,\"clock_domain\":\"gpu-local\","
    << "\"timer_unit\":\"ns\",\"timer_resolution_ns\":null,\"dropped_records\":0,"
    << "\"origin\":\"educational_two_role_kernel\",\"production_gate_eligible\":false,"
    << "\"trace_on_off_parity\":\"bitwise_pass\",\"build_verification\":\"not_integrated_with_artifact_manifest\","
    << "\"device_index\":" << device << ",\"compute_capability\":\"" << prop.major << '.' << prop.minor
    << "\",\"coverage\":\"all_demo_CTAs_one_lane_per_role\",\"ranges\":[\n";
  bool first = true;
  for (const auto& r : ranges) {
    if (!r.valid) throw std::runtime_error("missing range record");
    if (!first) f << ",\n";
    first = false;
    f << "{\"device_id\":\"cuda:" << device << "\",\"launch_id\":1,\"replay_id\":0,"
      << "\"cta\":[" << r.cta << ",0,0],\"warp_id\":" << r.warp
      << ",\"role\":\"" << (r.warp == 0 ? "cuda_role" : "tensor_role")
      << "\",\"stage\":\"" << (r.warp == 0 ? "scalar_math" : "wmma_scope")
      << "\",\"semantic\":\"software_scope\",\"token\":0,\"sm_begin\":" << r.begin_sm
      << ",\"sm_end\":" << r.end_sm << ",\"start_ns\":" << r.begin_ns
      << ",\"end_ns\":" << r.end_ns << '}';
  }
  f << "\n]}\n";
  if (!f) throw std::runtime_error("failed writing output");
  std::cout << "Wrote " << output << "; parity passed. This is NOT a performance qualification.\n";
}

__global__ void timer_observations(std::uint64_t* stamps) {
  if (threadIdx.x == 0)
    for (int i = 0; i < 1024; ++i) stamps[i] = flash_trace::read_stamp().ns;
}

static void calibrate(const std::string& output) {
  constexpr int blocks = 512, iterations = 128, repeats = 100, inner = 10;
  Stream stream;
  DeviceBuffer<half> a(256), b(256);
  DeviceBuffer<float> scalar(blocks * 32), tensor(blocks * 256);
  DeviceBuffer<flash_trace::RangeRecord> records(blocks * 2);
  DeviceBuffer<std::uint64_t> stamps(1024);
  std::vector<half> ones(256, __float2half(1.0f));
  CUDA_CHECK(cudaMemcpy(a.p, ones.data(), 256*sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.p, ones.data(), 256*sizeof(half), cudaMemcpyHostToDevice));
  timer_observations<<<1, 32, 0, stream.s>>>(stamps.p);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream.s));
  const auto timer = copy_to_host(stamps);
  auto launch = [&](int mode) {
    if (mode == 0)
      two_role_kernel<false><<<blocks,64,0,stream.s>>>(a.p,b.p,scalar.p,tensor.p,nullptr,iterations);
    else if (mode == 1)
      two_role_kernel<true><<<blocks,64,0,stream.s>>>(a.p,b.p,scalar.p,tensor.p,records.p,iterations);
    else
      two_role_kernel<true,true><<<blocks,64,0,stream.s>>>(a.p,b.p,scalar.p,tensor.p,records.p,iterations);
    CUDA_CHECK(cudaGetLastError());
  };
  std::vector<float> reference_scalar, reference_tensor;
  cudaGraph_t graph[3];
  cudaGraphExec_t executable[3];
  for (int mode = 0; mode < 3; ++mode) {
    for (int i = 0; i < 50; ++i) launch(mode);
    CUDA_CHECK(cudaStreamSynchronize(stream.s));
    const auto s = copy_to_host(scalar), t = copy_to_host(tensor);
    if (mode == 0) { reference_scalar = s; reference_tensor = t; }
    else if (std::memcmp(s.data(),reference_scalar.data(),s.size()*sizeof(float)) ||
             std::memcmp(t.data(),reference_tensor.data(),t.size()*sizeof(float)))
      throw std::runtime_error("off/coarse/focused parity failed");
    CUDA_CHECK(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeGlobal));
    for (int i = 0; i < inner; ++i) launch(mode);
    CUDA_CHECK(cudaStreamEndCapture(stream.s,&graph[mode]));
    CUDA_CHECK(cudaGraphInstantiate(&executable[mode],graph[mode],nullptr,nullptr,0));
  }
  cudaEvent_t begin,end;
  CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end));
  std::ofstream f(output);
  if (!f) throw std::runtime_error("cannot open calibration output");
  f << "{\"synthetic\":false,\"diagnostic_only\":true,\"blocks\":512,\"iterations\":128,"
       "\"inner\":10,\"repeats\":100,\"parity\":\"bitwise_pass\","
       "\"clock_policy\":\"unlocked\",\"timer_observations_ns\":[";
  for (std::size_t i=0;i<timer.size();++i) { if(i) f << ','; f << timer[i]; }
  f << "],\"samples\":[";
  bool first = true;
  // Fixed balanced order, selected before seeing any samples.
  const int orders[6][3]={{0,1,2},{2,1,0},{1,0,2},{2,0,1},{0,2,1},{1,2,0}};
  for (int block=0;block<6;++block) {
    for (int order=0;order<3;++order) {
      int mode=orders[block][order];
      for (int i=0;i<5;++i) CUDA_CHECK(cudaGraphLaunch(executable[mode],stream.s));
      CUDA_CHECK(cudaStreamSynchronize(stream.s));
      for (int i=0;i<repeats;++i) {
        CUDA_CHECK(cudaEventRecord(begin,stream.s));
        CUDA_CHECK(cudaGraphLaunch(executable[mode],stream.s));
        CUDA_CHECK(cudaEventRecord(end,stream.s));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms=0;
        CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
        if(!first) f << ',';
        first=false;
        f << "{\"block\":" << block << ",\"mode\":" << mode << ",\"us\":" << ms*1000/inner << '}';
      }
    }
  }
  f << "]}\n";
  if(!f) throw std::runtime_error("failed writing calibration");
  CUDA_CHECK(cudaEventDestroy(begin)); CUDA_CHECK(cudaEventDestroy(end));
  for(int mode=0;mode<3;++mode) {
    CUDA_CHECK(cudaGraphExecDestroy(executable[mode])); CUDA_CHECK(cudaGraphDestroy(graph[mode]));
  }
}

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--calibrate") { calibrate(argv[2]); return 0; }
    if (argc > 4) throw std::runtime_error("usage: profile_example [raw.json] [blocks] [iterations]");
    const auto out = argc > 1 ? std::string(argv[1]) : std::string("trace.raw.json");
    const int blocks = argc > 2 ? std::stoi(argv[2]) : 64;
    const int iters = argc > 3 ? std::stoi(argv[3]) : 128;
    if (blocks < 1 || blocks > 65536 || iters < 1 || iters > 65536)
      throw std::runtime_error("blocks and iterations must be in [1,65536]");
    profile(out, blocks, iters);
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
  return 0;
}
