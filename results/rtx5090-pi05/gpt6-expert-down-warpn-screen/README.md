# Expert FFN down: warp N=16 CPU screen

Decision: reject the strict warp-N-only edit because the existing vendor output
thread map produces a zero row-iteration count with epilogue kCount=8.
The conditional warpN=16 plus a candidate-specific kCount=4 passes the inspected
source-level shape/mapping constraints. It is a plausible single bounded local
screen, not a measured speedup or a compile-validated kernel. No kernel was
implemented, compiled, or launched; no Torch/model import was performed.

Inspected main source revision: 86f58f3. Site:
action_expert_ffn_down_residual, M=50, N=1024, K=4096, contiguous BF16 A/B/C/D,
gate[1024] broadcast over rows. Existing rounded epilogue performs fully reduced
FP32 accumulator -> BF16 -> FP32, separate __fmul_rn and __fadd_rn with gate and
residual, then BF16 store. The existing destination += fragment fix is required.

## Measurement evidence and limits

The measured/unit-mma.md sustained BF16/FP32 table contains 4, 8, and 12
warps/SM: 507.5, 511.5, and 511.5 FLOP/cycle/SM. mma_clock.cu actually loops over
{4,8,12}. mma_unit.cu's run_issue<1,2,4,8,16> varies independent accumulator sets
inside one warp; its sustained sweep also starts at 4 warps. There is no
sustained 1- or 2-warp result here. The single-warp 32.13-cycle instruction
measurement and the 4-warp ceiling do not measure the production 2-warp CTA's
tensor utilization or prove that doubling its warps doubles throughput.

The previous M16 experiment is separate and remains rejected: its actual
reset-inclusive 180-call ABBA medians were A 10.5696, B 13.0772, B 13.0844,
A 10.5819 us/call. Both traces reported grid 298/block 64. That experiment
changed CTA M and warp M; it neither tests nor disproves the present conditional
warpN/epilogue candidate. The recorded cfg9 compiler output used 120 registers
per thread; accumulator arithmetic below is only one part of that total.

Existing 12 EpiLinear configs do not include warpN=16. Configs 8 and 11 have
4 warps but change CTA shape/stages, and the old 18-weight GEMM sweep used an
EpiLinear output, not the current rounded gate/residual epilogue. They are not an
isolated 2-versus-4-warp result.

Evidence:
- src/flash_vla/hardware/nvidia/rtx5090/measured/unit-mma.md
- lab/sm120/mma_clock.cu:290 and lab/sm120/mma_unit.cu:296
- src/flash_vla/hardware/nvidia/rtx5090/pi0/backends/cuda/kernels/cutlass_gemm.cu:85
- results/pi05-rtx5090/gpt6-run-01/measurements/expert-down-cutlass-m16-rejected.json
- results/pi05-rtx5090/gpt6-run-01/measurements/expert-down-cutlass-m16-resources.txt
- results/pi05-rtx5090/gpt6-run-01/measurements/expert-cutlass-screen.json

## Strict edit: a source-level epilogue contradiction

DownDefaults uses CTA32x64x32, warp32x32x32, instruction16x8x16, 8 stages,
RoundedGatedResidual::kCount=8. Changing only warpN to 16 gives warp grid 1x4x1
and 128 threads instead of 1x2x1 and 64.

The actual template chain is DefaultGemmWithBroadcast ->
DefaultGemmUniversal -> DefaultGemm<Sm80> -> DefaultEpilogueTensorOp, with
EpilogueOutputOp::kCount passed unchanged as ElementsPerAccess. Its
DefaultThreadMapTensorOp supplies Shape=(64,8,1,1,1),
Count=(1,4,1,1,4), and 64 or 128 threads to OutputTileOptimalThreadMap.
No default silently narrows the vector.

For RowArrangement<...,true>, the vendor's preferred access is 256 bytes:

| Derived value | cfg9: 2 warps, count8 | strict: 4 warps, count8 | conditional: 4 warps, count4 |
|---|---:|---:|---:|
| ShapeRow = 8 / warps | 4 | 2 | 2 |
| ShapeWidth = 64 / count | 8 | 8 | 16 |
| TargetAccessRows | 2 | 2 | 1 |
| AccessWidth, lanes | 8 | 8 | 16 |
| AccessRows | 4 | 4 | 2 |
| IterationsColumn | 1 | 1 | 1 |
| IterationsRow = ShapeRow / AccessRows | 1 | **0** | 1 |

Thus the strict edit reaches static_assert(kIterationsRow > 0,
"Iteration Count Row must be > 0"). This conclusion follows the installed
template arithmetic; it is not a captured compiler diagnostic. Stop that
strict candidate without compilation.

Relevant vendor sources under third_party/cutlass/include/cutlass:
- gemm/kernel/default_gemm_with_broadcast.h:101
- gemm/kernel/default_gemm.h:352
- epilogue/threadblock/default_epilogue_tensor_op.h:546
- epilogue/threadblock/default_thread_map_tensor_op.h:55
- epilogue/threadblock/output_tile_thread_map.h:221

## Conditional count4: coverage and reduction remain structurally valid

A candidate-specific output operator with kCount=4 makes all fragment typedefs
and numerical converters four-wide; the scalar BF16/FP32 rounding sequence need
not change. It must not alter the control operator. BF16 output/FP32 accumulator
uses generic DefaultIteratorsTensorOp for both count8 and count4, with one
fragment per epilogue iteration.

With count4, thread (warp w, lane l) starts at
row=2*w+floor(l/16), column=4*(l%16), and handles four columns.
This covers every cell in one 8x64 slab exactly once. Four fragment indices
advance by 8 rows and cover 32x64. A CPU enumeration of these source formulas
found 512 unique cells/slab and 2048 unique cells/tile, all multiplicity one.
Applying the existing M50 row predicate across 2x16 output CTAs gives exactly
50x1024=51200 cells, all multiplicity one. N1024 and row strides are divisible
by four; reducing BF16 vector accesses from 16 to 8 bytes introduces no tail
vector or stricter alignment requirement.

TensorOpPolicy still has kAccumulatorFragments=(32/16)*(16/8)=4. Each thread's
fragment shrinks from 8 to 4 FP32 values, while block threads increase 64->128.
The complete accumulator remains 2048 FP32 values/CTA (8192 bytes). Each
Stream-K peer slot still reserves two tiles: 16384 bytes. The share/reduce
templates use the same candidate BlockStriped<128,Array<float,4>>, so their
indexing is mutually consistent. The byte interpretation changes from the
control's BlockStriped<64,Array<float,8>>; equal size does not make the two
workspace layouts interchangeable. The candidate needs its own matching
kernel/plan and the existing destination-fragment offset fix. No new reduction
scheme is indicated by this analysis.

The shared warp store spans 8 x (4*16) FP32 elements, and the output shared
loader reads the same 8x64 slab in four-wide vectors. Both its fragment and the
output fragment contain four elements, satisfying the inspected broadcast
epilogue assertions. Its padded epilogue shared buffer is still
8*(64+8)*4=2304 bytes.

Additional vendor sources:
- epilogue/threadblock/default_epilogue_tensor_op.h:99 (generic BF16/FP32 case)
- epilogue/warp/tensor_op_policy.h:66
- epilogue/warp/tile_iterator_tensor_op.h:92
- epilogue/threadblock/epilogue_base_streamk.h:70
- epilogue/threadblock/epilogue_with_broadcast.h:1628 (single-source reduction)
- epilogue/threadblock/predicated_tile_iterator.h:723
- src/flash_vla/hardware/nvidia/rtx5090/pi05/backends/cutlass_backbone.cu:38

## Mainloop scale, possible value, and missing evidence

The inspected row-major BF16 TensorOp specialization in
gemm/threadblock/default_mma_core_sm80.h:1670 has zero A/B padding. CTA/stages
stay fixed, so mainloop shared storage from mma_base.h:140 remains
(32*32 + 32*64)*8*2=49152 bytes. The full GEMM has 32 output tiles and 128
K32 iterations per output tile. Global operand tile volume does not decrease.

The new warp shape is divisible by the 16x8x16 MMA shape. The A/B cooperative
load maps remain positive: A per-thread 128-bit access iterations 2->1, B 4->2.
The B warp iterator's logical N16 is divisible by its N8 instruction and
8-element LDSM access; its congruous iterator has one LDSM iteration instead
of two. No additional source-level divisibility failure was found on this
path. This is not full template instantiation or PTX/SASS validation.

Each warp holds 16 instead of 32 FP32 accumulator elements/thread. Per K32 tile,
MMA instructions/warp become 8 instead of 16, but four instead of two warps keep
total CTA MMA work at 32 instructions. A's same warp fragment is now needed by
four N partitions, increasing replicated warp-level A fragment consumption;
B fragment width per warp halves. Fewer accumulators and more participating
warps might improve scheduling, but changed load distribution, smaller A reuse,
and twice as many half-width epilogue accesses can offset that benefit.

The Stream-K algorithm stays the same and its accumulator fragment count stays
four. Its runtime schedule also takes occupancy as input. A grid of 298 alone,
including an apparent 170 SK + 128 reduction decomposition, cannot establish
which SMs execute those blocks, their overlap, or active warps over time. Do not
assume one SK CTA per SM or attribute the M16 rejection to such a mapping.
New register allocation, actual occupancy, scheduling, and memory/tensor
utilization are unmeasured.

The conditional variant is small enough for one future numerical check followed
by the same actual 180-call reset-inclusive ABBA if authorized. It jointly
changes warp partitioning and output vector granularity; any gain must be
attributed to that joint candidate. There is no numerical speedup prediction,
no claim of bitwise equivalence, and no request to expand the tile sweep.
