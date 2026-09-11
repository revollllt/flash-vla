// Does a thread-block cluster actually launch, synchronise and share memory on
// this device? ptxas assembles `barrier.cluster` and `mapa.shared::cluster` for
// sm_120, which says nothing about whether a cluster of 2 can be placed. This
// probe launches one, has every CTA write into rank 0's shared memory through a
// DSMEM pointer, barriers, and checks rank 0 read what the others wrote.
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CHECK(expr)                                                            \
  do {                                                                         \
    cudaError_t e_ = (expr);                                                    \
    if (e_ != cudaSuccess) {                                                    \
      printf("    %-28s -> %s\n", #expr, cudaGetErrorString(e_));               \
      return false;                                                             \
    }                                                                           \
  } while (0)

// Each CTA writes its rank into rank 0's shared array, then rank 0 sums it.
__global__ void dsmem_sum(int *out, int expected_ctas) {
  extern __shared__ int tile[];
  cg::cluster_group cluster = cg::this_cluster();
  const unsigned rank = cluster.block_rank();

  if (threadIdx.x == 0) tile[rank] = 0;
  cluster.sync();

  if (threadIdx.x == 0) {
    // Map rank 0's shared memory into this CTA's address space (DSMEM).
    int *remote = cluster.map_shared_rank(tile, 0);
    atomicAdd(remote, static_cast<int>(rank) + 1);
  }
  cluster.sync();

  if (rank == 0 && threadIdx.x == 0) {
    int sum = tile[0];
    int want = expected_ctas * (expected_ctas + 1) / 2;
    out[0] = (sum == want) ? sum : -sum;
  }
}

static bool try_cluster(int cluster_dim, int threads, int smem_bytes) {
  int *d_out = nullptr;
  CHECK(cudaMalloc(&d_out, sizeof(int)));
  CHECK(cudaMemset(d_out, 0, sizeof(int)));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(cluster_dim, 1, 1);
  cfg.blockDim = dim3(threads, 1, 1);
  cfg.dynamicSmemBytes = smem_bytes;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = cluster_dim;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;

  cudaError_t launch = cudaLaunchKernelEx(&cfg, dsmem_sum, d_out, cluster_dim);
  if (launch != cudaSuccess) {
    printf("  cluster %-2d  LAUNCH FAILED: %s\n", cluster_dim,
           cudaGetErrorString(launch));
    cudaFree(d_out);
    cudaGetLastError();
    return false;
  }
  cudaError_t sync = cudaDeviceSynchronize();
  if (sync != cudaSuccess) {
    printf("  cluster %-2d  SYNC FAILED:   %s\n", cluster_dim,
           cudaGetErrorString(sync));
    cudaFree(d_out);
    cudaGetLastError();
    return false;
  }
  int h_out = 0;
  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  int want = cluster_dim * (cluster_dim + 1) / 2;
  printf("  cluster %-2d  ok, DSMEM sum = %d (want %d) %s\n", cluster_dim, h_out,
         want, h_out == want ? "PASS" : "*** MISMATCH ***");
  return h_out == want;
}

int main() {
  int dev = 0;
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev);
  printf("device: %s  cc %d.%d  %d SMs\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);

  int cluster_supported = 0;
  cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch, dev);
  printf("cudaDevAttrClusterLaunch            = %d\n", cluster_supported);

  int smem_optin = 0;
  cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  printf("cudaDevAttrMaxSharedMemoryPerBlockOptin = %d B\n", smem_optin);

  int dsmem = 0;
  cudaError_t e = cudaDeviceGetAttribute(
      &dsmem, cudaDevAttrClusterLaunch, dev);
  (void)e; (void)dsmem;

  int max_cluster = 0;
  cudaLaunchConfig_t occ_cfg = {};
  occ_cfg.gridDim = dim3(170, 1, 1);
  occ_cfg.blockDim = dim3(256, 1, 1);
  occ_cfg.dynamicSmemBytes = 1024;
  cudaError_t occ_rc = cudaOccupancyMaxPotentialClusterSize(
      &max_cluster, (void *)dsmem_sum, &occ_cfg);
  printf("cudaOccupancyMaxPotentialClusterSize    = %d (%s)\n", max_cluster,
         cudaGetErrorString(occ_rc));
  cudaGetLastError();

  printf("\nlaunching clusters (256 threads, 1 KB smem):\n");
  for (int dim : {1, 2, 4, 8, 16}) {
    try_cluster(dim, 256, 1024);
  }

  printf("\nshared-memory opt-in ladder (dynamic smem per block):\n");
  for (int kb : {48, 64, 99, 100, 128, 164, 227}) {
    cudaError_t rc = cudaFuncSetAttribute(
        dsmem_sum, cudaFuncAttributeMaxDynamicSharedMemorySize, kb * 1024);
    printf("  %3d KB  cudaFuncSetAttribute -> %s\n", kb,
           rc == cudaSuccess ? "accepted" : cudaGetErrorString(rc));
    cudaGetLastError();
  }
  return 0;
}
