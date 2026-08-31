#define __host__
#define __device__
#include "../../../src/flash_vla/hardware/nvidia/h100/pi05/backends/cuda/kernels/sm90_attn_task_desc.cuh"
#include <cstdio>
using namespace flash_vla::pi05::sm90::attn;
int main() {
  static_assert(QKV_TASKS == 80 && ATTN_TASKS == 64 && OUT_TASKS == 128);
  static_assert(QKV_TASKS + ATTN_TASKS + OUT_TASKS <= N_CTAS * TASK_SLOTS);
  static_assert(KEYS_PAD % ATTN_SPLIT == 0 && (KEYS_PAD / ATTN_SPLIT) % ATTN_BKK == 0);
  static_assert(QKV_W % QKV_BN == 0 && D % OUT_BN == 0 && D % QKV_SPLIT == 0);
  static_assert(CounterMap::kOutBegin + OUT_TILES <= CounterMap::kCount);
  static_assert(H * DH + 2 * DH == QKV_W);
  printf("tasks   qkv %d  attn %d  out %d   total %d of %d slots\n",
         QKV_TASKS, ATTN_TASKS, OUT_TASKS,
         QKV_TASKS + ATTN_TASKS + OUT_TASKS, N_CTAS * TASK_SLOTS);
  printf("trips   qkv %d  attn %d  out %d\n", QKV_TRIP, ATTN_TRIP, OUT_TRIP);
  printf("smem    %d B of 232448   (%.0f KB spare)\n", SMEM_B, (232448.0-SMEM_B)/1024);
  double f32 = 4, bf = 2;
  double wq = (QKV_PARTIAL_F32*f32), wa = (ATTN_PARTIAL_BF16*bf),
         wl = (ATTN_LSE_F32*f32), wo = (OUT_PARTIAL_F32*f32);
  printf("workspace  qkv_partial %.2f MB  attn_partial %.2f MB  lse %.0f KB  out_partial %.2f MB\n",
         wq/1e6, wa/1e6, wl/1024, wo/1e6);
  printf("           q_buf/o_buf %.0f KB each; total %.2f MB\n",
         H*M_PAD*DH*bf/1024, (wq+wa+wl+wo+2.0*H*M_PAD*DH*bf)/1e6);
  double traffic = 2*(wq+wa+wl+wo);
  printf("reduction traffic %.2f MB  -> %.2f us at L2 6447 GB/s\n",
         traffic/1e6, traffic/6447e9*1e6);
  return 0;
}
