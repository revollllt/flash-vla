# One full-row backbone attention candidate

CPU-only preparation against source 754d7f4. No model, Torch import, JIT,
NVCC, or GPU was run. Performance and final compiler resource use are unknown.

## Hypothesis and fixed geometry

Preserve the deployed BF16 QK score and BF16 probability rounding boundaries,
but keep both intermediates within a CTA that owns the entire key row and all
256 output channels. This may improve the two GEMMs and remove intermediate
global traffic. It is not online softmax: every row's max and sum include all
968 keys before any probability is rounded to BF16 and used in PV.

The only candidate is M32, N1024, QK BK32, PV BK64/D256, 8 warps, one stage.
There are exactly 242 CTAs (7744/32), with no query/output-channel padding.
M16/4-warps would have 484 CTAs. Both have a lower bound of 128 FP32 score
accumulators per thread; M32 halves full K/V matrix reads across CTAs, so it is
the selected bounded tradeoff. There is no alternative tile or autotune path.

Source and existing compiled evidence:

- The owning prior analysis is lab/pi05/rtx5090_backbone_attention.md.
  Its actual compiler module shows BF16 QK -> FP32 scale0.0625/mask/softmax ->
  BF16 P -> FP32-accumulating BF16 GEMM -> BF16 output copy.
- The deployed wrapper remains torch_ops.llm_backbone_attention, calling the
  existing compiled reference. Backbone attention occurs 17 times; the final
  layer emits only K/V.
- pipeline.py allocates separate llm_backbone_q and llm_backbone_attn buffers,
  both (7744,256), reused across layers. K/V are per-layer prefix cache views
  (968,256), row stride256, and mask is (968). Out does not alias Q.
- Installed Triton 3.7.1 language/core.py and semantic.py support BF16 dot with
  FP32 accumulation and rank2 gather from (32,1024) to (32,64). The local
  AccelerateMatmul.cpp explicitly selects MMA v2 for compute capability120.
  The deployed expert QK already executes this BF16 MMA path on SM120.
  These establish API/architecture expressibility, not a compiled resource
  guarantee for this much larger tile.

## Resources and layout boundaries

QK iterates eight BK32 tiles. K operand per stage is 32*1024*2 = 65,536 B;
Q operand is 32*32*2 = 2,048 B. The ideal staged operand total is 66 KiB.
BK64 would require 128 KiB for K alone and is excluded. One stage is explicit;
no full K256 operand is requested.

QK scores are 32*1024 FP32 = 128 KiB of distributed accumulator payload,
128 scalar FP32 values per thread at 256 threads. This is register data, not
a claim that 128 KiB fits shared. BF16 probability payload is 64 KiB total:
64 registers/thread if packed two values/register, potentially128 if unpacked.
PV output accumulation is32*256 FP32 =32 KiB,32 registers/thread, live with P.
A PV step has4 KiB of P and32 KiB of V operands. These payload lower bounds
exclude address registers, softmax temporaries, shuffle duplication, layout
conversion scratch, and compiler scheduling. Actual registers/spills/shared
must be inspected before actual-model capture.

The local chained-dot warp heuristic prefers warps [1,8] when M<N, but the
two dots sit in separate loops; final layouts must be read from compiler IR.
If that partition is retained, each QK warp owns a32x128 score strip. Full-row
reductions then exchange partials across warps. No physical CTA-to-SM placement
is assumed. 242 CTAs can expose170 SMs; the later72 CTAs are a logical tail,
not proof of occupancy or dispatch placement.

The entire P is normalized exactly once per CTA. PV gathers sixteen consecutive
64-column pieces. GatherOpToLLVM.cpp explicitly supports shared-memory fallback:
store the source P, CTA barrier, indexed shared loads. In that path it may
rewrite64 KiB of P each iteration (1 MiB/CTA,242 MiB/call), even though softmax
is not repeated. If the compiler makes the gather warp-local, its alternative
is shuffle/select work; neither path is assumed cheap. Gather scratch is scoped
to that operation, whereas PV operand staging occurs later;66 KiB QK,64 KiB
gather and36 KiB PV are phase payloads, not amounts that can safely be summed or
a promise of full reuse. A layout conversion could independently exceed shared
capacity or register limits. A nontrivial layout/resource failure stops this
fixed candidate; it does not trigger a tile search.

## Work and memory estimate

The padded QK and PV each use2*7744*1024*256 FLOPs,8.120 GFLOP combined,
5.79% above the unpadded968-key arithmetic. There are8,192 logical
m16n8k16 instructions per CTA across the two GEMMs (4,096 each), before any
compiler replication. Every key is normalized once per query; no8x softmax
duplication as in the rejected expert N32 output split.

A K or V matrix is495,616 B. M32 reads each matrix242 times in logical CTA
traffic:119,939,072 B each,239,878,144 B combined per call. M16 would double
that. Most reuse may hit cache, but there is no NCU evidence establishing its
hit rate or attained L2 bandwidth. Do not price it at DRAM bandwidth.

Eliminating global BF16 score/P storage removes four traversals of
7744*968*2 =14,992,384 B,59,969,536 B/call. The standalone final copy adds
7,929,856 B of read+write traffic. Repeated CTA K/V reads and gather/shared
work can outweigh those savings; local speedup is unknown.

The existing profile attribution is approximately1.122 ms/17 calls:
QK0.525, PV0.431, softmax0.133, copy0.033 ms. Removing just softmax/copy has
only0.166 ms of total attributed time. The hypothesis must win across the
complete two-GEMM chain; kernel sums are not deployed latency.

## Fixed execution protocol after GPU grant

1. resources: compile this exact candidate on zero tensors; save PTX,
   registers, spills and shared bytes. This is not numerical validation.
2. prepare: build shipped once with belt-cup and seed42; sample_inputs(42);
   replay the real vision stage and host slot, then instrument the actual
   backbone eager calls. Capture all17 pre-attention Q/K/V/mask plus original
   outputs. Record actual shapes, strides and Q/out addresses. The original
   outputs continue into subsequent layers; no candidate value is fed back.
3. check: compare candidate with all17 captured actual outputs, and verify the
   current compiled control against those outputs. Use the existing shallow
   rel_rms/cosine thresholds unchanged. Report valid query rows separately,
   all-row metrics, and all-row finiteness. Stop at the first failing layer;
   preserve metrics/traceback, with no tile sweep. Reduction order may change;
   neither BF16 boundary implies bitwise equality.
4. time: only after successful check, run one A/B/B/A of all17 full calls.
   Both routes use the same common Q/out allocations and identical per-layer
   K/V/mask addresses. Both copy the same saved Q into the common Q buffer
   inside timing. Q and out are distinct. Both overwrite the complete out;
   neither gets an extra or omitted output reset.
   CUDA events on a fresh capture stream each leg, four17-call chains per
   graph, warm5/repeat30. Repeat the actual working set without L2 flushing
   or locked clocks; this is not a claim of cold-cache performance.
   Preserve all120 raw samples. No further rounds or production integration
   if the gain fails to separate from within-route ABBA drift.

Run with the usual environment and PYTHONPATH, using:

    python -m lab.pi05.backbone_fullrow resources --output "$RESOURCES"
    python -m lab.pi05.backbone_fullrow prepare --snapshot "$SNAPSHOT" \
      --checkpoint "$CHECKPOINT" \
      --checkpoint-id kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha
    python -m lab.pi05.backbone_fullrow check --snapshot "$SNAPSHOT" --output "$CHECK"
    python -m lab.pi05.backbone_fullrow time --snapshot "$SNAPSHOT" --output "$TIMING"

CPU checks: python3 -m py_compile lab/pi05/backbone_fullrow.py and targeted
source/shape arithmetic review. Production, registry and routing are untouched.
