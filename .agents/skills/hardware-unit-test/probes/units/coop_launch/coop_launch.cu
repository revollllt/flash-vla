// coop_launch -- the hut ABI over the cooperative-launch probe.
//
// The launch path is `cudaLaunchCooperativeKernel`, not `<<<>>>`: it is the only
// legal launch for a kernel calling grid_sync, and it REFUSES a grid that
// cannot be co-resident rather than deadlocking. Both facts are why this unit
// can enumerate the placement limit instead of assuming it.

#include "coop_launch.cuh"

namespace cl = hut::coop_launch;

// opt[] carries what is not a launch dimension.
enum : int32_t {
  OPT_N_ITERS = 0,     // barriers (or loop trips) inside one launch
};

extern "C" {

const char* hut_name() { return "coop_launch"; }

uint32_t hut_flags() {
  // No COLD: nothing here touches memory. No CHECK: the thing measured is a
  // barrier, and a barrier that failed to synchronise HANGS or trips the
  // cooperative launch check -- it does not return wrong bytes.
  return HUT_REPORTS_SM;
}

int32_t hut_cfg_count() { return 2; }

int32_t hut_cfg(int32_t cfg, int32_t field) {
  if (cfg < 0 || cfg > 1) return HUT_ERR_BAD_CFG;
  return field == 0 ? cfg : HUT_ERR_BAD_CFG;   // cfg IS the mode
}

const char* hut_cfg_name(int32_t field) {
  return field == 0 ? "mode" : nullptr;        // 0 grid_sync, 1 empty
}

const char* hut_opt_name(int32_t i) {
  return i == OPT_N_ITERS ? "n_iters" : nullptr;
}

int32_t hut_smem(const HutParams*) { return 0; }

// Does this device support cooperative launch at all, and how many blocks of
// this shape can be co-resident? Reported rather than assumed: the launch
// unit's `cluster.count.max` records that placement follows the occupancy
// query, not the SM count, and the same holds here.
int32_t hut_max_blocks(int32_t n_threads) {
  int32_t supported = 0, dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return HUT_ERR_BAD_PARAM;
  if (cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, dev)
      != cudaSuccess || !supported)
    return 0;
  int32_t per_sm = 0;
  cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm, reinterpret_cast<const void*>(cl::grid_sync_kernel), n_threads, 0);
  if (e != cudaSuccess) return static_cast<int32_t>(e);
  int32_t sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  return per_sm * sms;
}

int32_t hut_launch(const HutParams* p, const HutBuffers* b, void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
  if (p->n_ctas < 1 || p->n_threads < 32 || p->opt[OPT_N_ITERS] < 1)
    return HUT_ERR_BAD_PARAM;

  auto* cycles = static_cast<long long*>(b->cycles_a);
  auto* sm_id = static_cast<int32_t*>(b->sm_id);
  auto* sink = static_cast<float*>(b->sink);

  if (p->mode == 1) {
    // The relaunch baseline runs through the ordinary launch path on purpose:
    // what it measures is what a caller pays TODAY, launch overhead included.
    cl::relaunch_kernel<<<p->n_ctas, p->n_threads, 0, s>>>(
        p->opt[OPT_N_ITERS], sink);
    return static_cast<int32_t>(cudaGetLastError());
    // (the launch above is the only thing that can fail here; the cooperative
    // path consumes its own refusals so they cannot surface at this line)
  }

  int32_t n_iters = p->opt[OPT_N_ITERS];
  int32_t mode = p->cfg;
  void* args[] = {&n_iters, &mode, &cycles, &sm_id, &sink};
  // Returns cudaErrorCooperativeLaunchTooLarge (720) when the grid cannot be
  // co-resident. Passed back unchanged so the host can record the limit.
  cudaError_t e = cudaLaunchCooperativeKernel(
      reinterpret_cast<const void*>(cl::grid_sync_kernel),
      dim3(p->n_ctas), dim3(p->n_threads), args, 0, s);
  // CONSUME the flag on refusal. A rejected launch also sets the context's
  // last-error, and the next `cudaGetLastError()` -- in the ordinary launch
  // path, which cannot produce a 720 of its own -- would read it back and
  // report someone else's failure as its own. Enumerating legality has to be
  // non-destructive, or the enumeration poisons everything measured after it.
  if (e != cudaSuccess) cudaGetLastError();
  return static_cast<int32_t>(e);
}

int32_t hut_check(const HutParams*, const HutBuffers*, void*) {
  return HUT_ERR_NO_CHECK;
}

}  // extern "C"
