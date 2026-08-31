#pragma once

// ABI for the Pi0.5 action-expert attention block as one persistent task loop:
// qkv projection -> multi-query flash attention -> gated output projection,
// for ONE layer-step.  Scope mirrors `sm90_ffn_task_desc.cuh`: a fixed 132-CTA
// worker grid walking a static task table baked offline by the host planner.
//
// This header is deliberately separate from the FFN's.  The two kernels share
// the task-loop IDIOM, not a binary ABI, so neither constrains the other's
// descriptor while both are under development.  Merge them only once both
// land.
//
// Geometry comes from the tile analysis, whose floors use measured machine
// constants (hardware-unit-test/sm90/constants.yaml) rather than datasheet
// peaks.  Each constant below names the tag that pins it.

#include <cstdint>
#include <type_traits>

namespace flash_vla::pi05::sm90::attn {

// ---------------------------------------------------------------- problem
constexpr int M = 50;          // action chunk
constexpr int M_PAD = 64;      // one wgmma m64 tile; rows 50..63 zeroed
constexpr int D = 1024;        // action-expert width
constexpr int H = 8;           // query heads
constexpr int DH = 256;        // head width
constexpr int QKV_W = 2560;    // H*DH (Q) + DH (K) + DH (V); multi-query
// COMPILE-TIME, deliberately.  The prompt is padded to MAX_TOKEN_LEN=200 and
// the pad is masked, so the only per-inference quantity is `n_valid`, which the
// additive key mask carries -- no shape depends on it.  Freezing PREFIX_LEN is
// what makes ATTN_TRIP a compile-time bound: a runtime KEYS would take the ring
// phase and the mainloop trip off the constant path, and wgmma.stages.wg.knee's "never
// wait_group 0 across the loop boundary" is worth 20-30% and hard to hold with
// a runtime trip.  The price is that keys [n_valid, PREFIX_LEN) are read and
// masked rather than skipped: ~65 of 1024 at a measured prompt, ~6%.  No split
// is ENTIRELY masked -- the hole sits inside the split that also holds the real
// suffix -- so a skip-if-masked early exit would recover none of it.
constexpr int PREFIX_LEN = 968;  // 3 views * 256 image tokens + 200 prompt

// KEYS is 1018 = PREFIX_LEN + M.  Padded to 1024 so ATTN_SPLIT divides it
// evenly and every split walks the same number of ring stages; the caller must
// extend the additive key mask to cover [1018, 1024) with MASK_NEG.  An
// unpadded 1018 makes one split short and reintroduces the ragged-tail branch
// `_num_splits` exists to avoid in the TileLang path.
constexpr int KEYS = PREFIX_LEN + M;
constexpr int KEYS_PAD = 1024;

// ------------------------------------------------------------ tile choice
// Every activation stays ROW-MAJOR.  Row-major caps boxDim[0] at 64 bf16
// elements (128 B under SW128, [tma.bytes.txn.max]); a 3-D box {64, rows, k/64}
// with the 64-element chunk as the OUTER dimension lifts BK past that in ONE
// TMA -- the box lands as [chunk][row][64], which is exactly the CuTe SW128
// K-major image -- so BK=256 costs one 32 KB copy per stage, not four.  The
// copy column is TMA count x 248 ns [tma.issue.warp]; ncu on the BK=64 kernel (job
// 555126) showed 11% DRAM, 4% SM, 91% no-eligible: issue-latency bound.
constexpr int QKV_BN = 64;     // 64 bf16 = 128 B box row, the SW128 maximum
// BK=128 x ring depth 4 (16 KB frames, 8 stages, 4 in flight) for the ONE-KERNEL
// qkv: BK=256 x depth 2 with no split exposed two full refills (job 556008,
// mainloop 7.9 us), and split-K 2 needs a second launch worth 2.7 us
// (job 556141) that a 6.2 us split kernel cannot amortise.
constexpr int QKV_BK = 128;    // one 3-D box {64, 64, 2} for x, one {64, 128} box for W
// No split-K: one kernel per op is the standalone target (the reduce launch
// costs 2.7 us, job 556141); the depth-4 ring above is what makes the 8-stage
// task viable.  In the task loop the 92 idle slot-0 CTAs host attention.
constexpr int QKV_SPLIT = 1;
constexpr int QKV_TILES = QKV_W / QKV_BN;              // 40
constexpr int QKV_TASKS = QKV_TILES * QKV_SPLIT;       // 40
constexpr int QKV_TRIP = (D / QKV_SPLIT) / QKV_BK;     // 8 stages

// One CTA owns one (head, kv-split).  Per-head rather than per-flat-row-block
// because a head's Q comes from exactly 4 qkv n-tiles, where a flat block of
// 64 rows spans all 8 heads and would depend on all 32 of them.
constexpr int ATTN_BMQ = M_PAD;   // all 50 tokens of one head, padded
// 64 keys per stage: at 32 the S = Q K^T wgmma (m64n32k16) re-reads the 2 KB Q
// A-tile per 1 KB of K and runs shared-memory-bound at ~75 cycles/instruction,
// 3x [wgmma.issue.wg.ss]; ablation job 555193 put 2.5 of the 5.1 us attention mainloop
// on it.  m64n64k16 halves the Q re-read per key [wgmma.issue.wg.ss 93% of peak].
constexpr int ATTN_BKK = 64;      // 32 KB K/V frames, ring depth 2 (whole split in flight)
constexpr int ATTN_DEPTH = 2;
constexpr int ATTN_SPLIT = 8;     // knee: below it per-CTA ingest binds
                                  // [tma.bw.cta.dram 133 GB/s], above it the KV
                                  // re-read across heads binds in L2
constexpr int ATTN_TASKS = H * ATTN_SPLIT;             // 64
constexpr int ATTN_TRIP = (KEYS_PAD / ATTN_SPLIT) / ATTN_BKK;  // 2 stages

// The kv-split combine is its own task kind, dealt across the CTAs that are
// idle while attention runs, one (head, 8-row group) each -- the shape of the
// TileLang fd_combine kernel.  Every split publishes a normalised partial and
// (m, l); a combine task reads 8 x 4 KB of contiguous rows.  Having split 0
// fold its siblings instead (v1-v7) put 224 KB through one CTA's TMA ceiling
// after the slowest sibling: 2.8 us join + 6.3 us combine on the critical path.
constexpr int COMBINE_ROWS = 8;
constexpr int COMBINE_TASKS = H * (M_PAD / COMBINE_ROWS);  // 64

// o_proj contracts over H*DH = 2048.  Its split count is not a tuning knob:
// SPLIT == H makes split `h` exactly head `h`'s contribution, so the
// dependency is "head h is combined" and nothing finer is needed.
constexpr int OUT_BN = 64;
constexpr int OUT_BK = 256;    // one head's k-slice per stage: o_buf[h] is one 3-D box
constexpr int OUT_SPLIT = H;                           // 8
constexpr int OUT_TILES = D / OUT_BN;                  // 16
constexpr int OUT_TASKS = OUT_TILES * OUT_SPLIT;       // 128
constexpr int OUT_TRIP = DH / OUT_BK;                  // 1 stage

// --------------------------------------------------------------- launch
constexpr int N_CTAS = 132;    // 1 x SM count; [sched.ctas.sm.knee] says a grid must
                               // reach 3x before extra CTAs/SM become warps, so
                               // 1 CTA/SM is the operating point regardless
constexpr int TASK_SLOTS = 3;  // qkv + attention | combine | o_proj

// One 128-thread math warpgroup plus two single-warp TMA producers, matching
// the FFN profile.  A second math warpgroup buys no tensor-core throughput
// [wgmma.ratio.sm.wg2] and the accumulator union fits: attention's S(64x32) plus
// O(64x256) is 152 registers per thread against the 255 cap.
constexpr int THREADS = 192;

// Static shared pool, sized by the widest body (attention).  qkv and o_proj
// need 64 KB, attention 160 KB, so ONE pool serves all three and no paging is
// required -- the mechanism the FFN spec predicted the full-decoder scope would
// need.  The trailing 1 KB holds the mbarrier pool.  Against the 232448 B
// per-CTA cap this leaves ~66 KB spare.
constexpr int SMEM_POOL_B = 163840;
constexpr int SMEM_B = SMEM_POOL_B + 1024;

// ----------------------------------------------------------------- tasks
enum class TaskKind : int32_t {
  kQkvProj = 0,       // n-tile of the QKV projection, + RoPE, + KV-cache store
  kAttention = 1,     // (head, kv-split) online-softmax pass
  kOutProj = 2,       // n-tile of the gated output projection
  kCombine = 3,       // (head, 8-row group) kv-split combine into o_buf
  kSentinel = -1,
};

// Four int32, binary-compatible in shape with the FFN descriptor so the host
// planner and the table validator can be shared.
//   kQkvProj    column = n-tile 0..39   dependency = counter to release
//                                       split  = 0..1
//   kAttention  column = head 0..7      dependency = q counter to await
//                                       split  = 0..7
//   kOutProj    column = n-tile 0..15   dependency = unused
//                                       split  = head 0..7
//   kCombine    column = head 0..7      dependency = attention counter to await
//                                       split  = row group 0..7
struct TaskDescriptor {
  TaskKind kind;
  int32_t column;
  int32_t dependency;
  int32_t split;
};

static_assert(std::is_same_v<std::underlying_type_t<TaskKind>, int32_t>);
static_assert(sizeof(TaskDescriptor) == 4 * sizeof(int32_t));

__host__ __device__ constexpr bool is_sentinel(TaskKind k) { return k == TaskKind::kSentinel; }
__host__ __device__ constexpr bool is_qkv(TaskKind k) { return k == TaskKind::kQkvProj; }
__host__ __device__ constexpr bool is_attention(TaskKind k) { return k == TaskKind::kAttention; }
__host__ __device__ constexpr bool is_out_proj(TaskKind k) { return k == TaskKind::kOutProj; }
__host__ __device__ constexpr bool is_combine(TaskKind k) { return k == TaskKind::kCombine; }

// The table is (N_CTAS, TASK_SLOTS, 4) int32; row c is CTA c's private list,
// executed in order, with kSentinel meaning idle.  Tasks are dealt one kind per
// slot so that every ring in a slot is monomorphic -- a type switch inside one
// CTA would change the shared-memory frame layout and force a drain at the
// boundary, losing exactly the ring continuity this kernel exists to keep:
//
//   slot 0  CTA   0..79   kQkvProj   (column, split) = divmod(c, QKV_SPLIT)
//   slot 0  CTA  80..131  kAttention tasks 0..51 (task i = (head, split) = divmod(i, 8))
//                         -- start at t=0: K/V frames inside the prefix are
//                         dependency-free and land while qkv runs; only the
//                         frame that overlaps the cache suffix waits on kKv,
//                         and Q waits on its head's counter
//   slot 1  CTA 1,3,..,23 kAttention tasks 52..63: the split-1 qkv CTAs only
//                         publish a partial and are free ~3 us before any Q
//                         counter flips, so they still prefetch
//   slot 1  CTA  24..87   kCombine   (column, split) = divmod(c - 24, M_PAD / COMBINE_ROWS)
//   slot 2  CTA   0..127  kOutProj   (column, split) = divmod(c, OUT_SPLIT)
// Attention splits need not be co-resident with each other: since the
// combine is its own task kind, no attention task waits on a sibling.
//
// The splits of one tile land on consecutive CTAs so they are co-resident,
// which is what makes split 0's wait on its siblings a poll rather than a
// deadlock.  Slot 0 leaves 52 CTAs idle and slot 1 leaves 68: there are only
// 40 qkv n-tiles and 8 heads to go around, and no tiling recovers that.  The
// answer is the full-decoder scope, where the previous layer's FFN tasks fill
// the same slots; it is not a defect of this table.

// -------------------------------------------------------------- counters
// Split 0 of every tile is the reducer, the pattern `ffn_taskloop.cu` already
// validates: splits 1..S-1 write a partial and bump a counter, split 0 waits
// for S-1 arrivals and folds them into its own accumulator before the epilogue.
// gmem counters are NOT barriers; release is `fence.release` + `red.global.add`
// [atom.ratio.ret: red is 1.30x atom], acquire is an `ld.global.acquire` poll.
// Measured release->acquire RTT is 640 ns median (job 541290), far under the
// 2 us threshold that would force a coarser schedule.
struct CounterMap {
  static constexpr int kQBegin = 0;       // [0,  8)  one per head: 4 qkv n-tiles arrive
  static constexpr int kKv = 8;           //           8 qkv n-tiles arrive (K and V)
  static constexpr int kAttnBegin = 9;    // [9, 17)  one per head: all 8 splits arrive
  static constexpr int kOBegin = 17;      // [17,25)  one per head: 8 combine tasks arrive
  static constexpr int kOutBegin = 25;    // [25,41)  one per o_proj tile: 7 splits arrive
  static constexpr int kQkvJoinBegin = 41;  // [41,81) one per qkv tile: split 1 arrives
  // Every qkv task arrives once its mainloop has retired, i.e. once its last
  // TMA read of `x` has landed.  o_proj split 0 waits for all QKV_TASKS before
  // it writes `out`, because `out` may alias `x` (the pipeline aliases them)
  // and a projection tile still streaming `x` would otherwise read the sum.
  static constexpr int kQkvDone = 81;
  // One flag per (head, kv-split) so split 0's producer stages each sibling's
  // partial as soon as THAT sibling has published, overlapping the copy with
  // the others' skew instead of waiting for the slowest before the first byte
  // moves (job 555209: 2.3-3.6 us join wait + 5.5 us combine on the critical
  // path).  kAttnBegin stays as the aggregate count for validation.
  // Standalone o_proj (one launch, last arriver reduces): its own tile
  // counters, never touched by the task loop, zero at allocation and reset
  // by the last arriver -- so no host memset and no cross-path residue.
  static constexpr int kSaOutBegin = 96;  // [96, 112)
  static constexpr int kCount = 160;      // host-zeroed before a task-loop launch

  static constexpr int kQArrive = 4;      // qkv tiles 4h..4h+3 produce head h's Q
  static constexpr int kKvArrive = 8;     // qkv tiles 32..39 produce K and V
  static constexpr int kAttnArrive = ATTN_SPLIT;
  static constexpr int kOArrive = M_PAD / COMBINE_ROWS;
  static constexpr int kOutArrive = OUT_SPLIT - 1;
  static constexpr int kQkvJoinArrive = QKV_SPLIT - 1;
  static constexpr int kQkvDoneArrive = QKV_TASKS;
};

// ------------------------------------------------------------- workspace
// Sizes the host must allocate.  Reduction traffic is a FIRST-ORDER cost at
// M=50, not a rounding detail: the three partial buffers move ~9 MB per
// layer-step between write and read, which is comparable to the 10.5 MB of
// weights.  Attention partials are bf16 for exactly that reason -- fp32 would
// double the largest of them.
// Element counts, not bytes.  Kept as plain int expressions so the python
// mirror can evaluate this header directly (`attn_reference.geometry`); the
// static_assert below is what guards the int32 arithmetic.
// Sized for every split (the standalone reduce kernels fold all of them);
// the task loop uses slots [0, S-1) and split 0 folds in registers.
constexpr int QKV_PARTIAL_F32 = QKV_SPLIT * QKV_TILES * M_PAD * QKV_BN;
constexpr int ATTN_PARTIAL_BF16 = ATTN_SPLIT * H * M_PAD * DH;  // row-major (s, h, row, dh)
constexpr int ATTN_LSE_F32 = ATTN_SPLIT * H * M_PAD * 2;         // (s, h, row, {m, l})
constexpr int OUT_PARTIAL_F32 = OUT_SPLIT * OUT_TILES * M_PAD * OUT_BN;
static_assert(ATTN_PARTIAL_BF16 > 0 && OUT_PARTIAL_F32 > 0, "workspace overflowed int32");

}  // namespace flash_vla::pi05::sm90::attn

// ------------------------------------------------------------------- ABI
// Every pointer is device memory.  Shapes are row-major unless noted.
// Returns 0 on success, a CUresult offset by 1000 for a descriptor failure, or
// a local code >= 1100 for an argument mismatch.
//
//   x           (M_PAD, D)        bf16  in   activation; rows 50..63 zeroed
//   rms_factor  (M_PAD,)          bf16  in   F = rsqrt(mean(x^2)+eps); computed
//                                            OUTSIDE, as in the FFN prototype
//   ada_scale   (D,)              bf16  in   s = 1 + scale, per (step, layer)
//   w_qkv       (D, QKV_W)        bf16  in   NOT pre-blocked: BN=64 already
//                                            gives a legal 128 B box row, and
//                                            [tma.bw.cta.geom] prices a gather-only
//                                            pre-block at ~3%
//   qkv_bias    (QKV_W,)          bf16  in   b = shift @ w_qkv
//   rope        (M_PAD, DH)       bf16  in   cos in even columns, sin in odd
//   key_mask    (KEYS_PAD,)       bf16  in   0 on real keys, MASK_NEG on the
//                                            prompt pad AND on [KEYS, KEYS_PAD)
//   w_o         (H*DH, D)         bf16  in
//   ada_gate    (D,)              bf16  in   g, per (step, layer)
//   k_cache     (KEYS_PAD, DH)    bf16  inout  rows [0,PREFIX_LEN) read-only;
//   v_cache     (KEYS_PAD, DH)    bf16  inout  [PREFIX_LEN,KEYS) written here
//   out         (M_PAD, D)        bf16  inout  in = residual, out = residual +
//                                              (attn @ w_o) * g.  MAY ALIAS `x`,
//                                              and does in the decoder pipeline,
//                                              where one buffer is both the
//                                              projection input and the residual.
//                                              Safe because the counter chain
//                                              orders every read of `x` before
//                                              any write of `out`: an o_proj task
//                                              waits on a head, a head waits on
//                                              every projection tile.
//   q_buf       (H, M_PAD, DH)    bf16  scratch  HEAD-MAJOR: makes qkv's store
//   o_buf       (H, M_PAD, DH)    bf16  scratch  and attention's load both
//                                                contiguous, and lets o_proj
//                                                read head h as one k-slice
//   qkv_partial / attn_partial / attn_lse / out_partial  see the sizes above
//   counters    (kCount,)         u32   inout  host-zeroed; replay-safe
//                                              self-reset is deferred, as in
//                                              the FFN prototype
// `timeline` is optional: a device (N_CTAS, TASK_SLOTS, 5) int64 buffer that
// receives %globaltimer stamps per task (slot start, first frame landed,
// mainloop retired, split join satisfied, task end) for critical-path analysis; null disables it.
// `n_ctas` and `prefix_len` are validated, not used: they exist so a caller
// built against a different shape profile fails loudly.  Reading the wrong KV
// rows produces a plausible wrong action rather than an error, which PLAN 4.4
// records as the expensive failure mode.  A new shape profile (num_views != 3,
// prompt_len != 200) is a recompile; migrate these constexprs to template
// parameters if a second profile ever has to coexist.
extern "C" int attn_taskloop_launch(
    const void* table, int n_ctas, int prefix_len,
    const void* x, const void* rms_factor, const void* ada_scale,
    const void* w_qkv, const void* qkv_bias, const void* rope,
    const void* key_mask, const void* w_o, const void* ada_gate,
    void* k_cache, void* v_cache, void* out,
    void* q_buf, void* o_buf,
    void* qkv_partial, void* attn_partial, void* attn_lse, void* out_partial,
    void* counters, void* dbg, void* timeline, void* stream);

// Standalone form of the same bodies: one ordinary grid kernel per op, the
// caller launches ops 0..5 in order (qkv split, qkv reduce, attention split,
// combine, o_proj split, o_proj reduce).  Same tensors, no table/counters
// (the pointer is accepted and ignored).  This is how one op's number is
// compared with the TileLang kernel for that op.
extern "C" int attn_standalone_launch(
    int op, int prefix_len,
    const void* x, const void* rms_factor, const void* ada_scale,
    const void* w_qkv, const void* qkv_bias, const void* rope,
    const void* key_mask, const void* w_o, const void* ada_gate,
    void* k_cache, void* v_cache, void* out,
    void* q_buf, void* o_buf,
    void* qkv_partial, void* attn_partial, void* attn_lse, void* out_partial,
    void* counters, void* dbg, void* timeline, void* stream);
