# One full-row backbone attention candidate

Initial CPU-only preparation used source754d7f4 (lab commit6e8d927).
The subsequent authorized zero-input resource observation is recorded below.
The later authorized actual-input check and sole ABBA are recorded below.

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

## Authorized resource observation

The sole candidate compiled and its zero-input launch completed successfully
on2026-09-14 at23:49:48.596–23:49:52.564 UTC. Invocation cwd was main at
cad1246, using the worker absolute script6e8d927 and main's existing environment.
Results are in results/rtx5090-pi05/gpt6-backbone-fullrow:
invocation.json, compile.log, resources.json, ptx-evidence.json, and the selected
original-numbered ptx-boundaries.txt. Full resources.ptx remains beside them
as an untracked generated artifact.

The reported resources are255 registers/thread, n_spills26, and67,584 bytes
shared. In installed Triton3.7.1 driver.c lines189–190, n_spills is
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES divided by4. The observation therefore
means104 B/thread local-memory footprint; it does not count spill accesses or
establish a latency penalty. Shared allocation is66 KiB, within the launch's
available capacity. GPU was released immediately after this resource stage.

Both BF16 boundaries survive compilation. After the QK loop's final MMA and
backedge, PTX700–827 explicitly rounds FP32 scores to BF16, and829–956 widens
back to FP32 before the separate scale and mask arithmetic. Full-row division
is at2061–2188;2190–2253 rounds normalized probabilities into BF16x2. These
packed probabilities pass through shared/layout/gather operations and BF16
ldmatrix loads into the PV mma.sync.f32.bf16.bf16.f32 instructions from6946.
This is PTX dataflow evidence, not an assertion based only on source casts.

The gather path includes4,096 static shfl.sync.idx instructions in PTX input;
this count precedes PTXAS optimization and does not establish executed SASS
count. Resource pressure warrants inspection but does not prove the candidate
is slower. No actual17-layer capture, correctness pass, or ABBA has run.
Continue only if the experiment owner authorizes the unchanged mapping.

## Actual-input result: fixed mapping rejected

The subsequent authorized capture used main c6a7d56, which differs from the
retained 024 revision 84f50c7 only by adding this lab source/documentation.
Capture metadata confirms all three masked backbone outproj/FFN routes were
bucketed-backbone; attention remained torch. There were 17 actual layer calls,
one shared Q address and one distinct shared out address. The seed42 input had
895 valid prefix keys. No candidate output was fed into capture.

All 17 candidate outputs passed the existing shallow tolerance, including
all-row finiteness. Worst rel_rms was 5.217102491502758e-5 and minimum cosine
was 0.9999999986391401. The unchanged control also passed against the captured
outputs. Detailed per-layer and valid-query-row metrics are in check.json.

The only ABBA measured the complete attention chain plus the identical Q reset:

| Leg | Median ms per 17 calls |
| --- | ---: |
| A1 control | 1.158411979675293 |
| B1 candidate | 1.7525280117988586 |
| B2 candidate | 1.7547000050544739 |
| A2 control | 1.1904480457305908 |

Candidate minus control was +0.5791839957 ms per 17 calls, or +34.06965 us/call.
Within-control drift was 0.0320360661 ms and within-candidate drift
0.0021719933 ms. Even the closest A/B medians were separated by 0.5620799661 ms.
The fixed mapping is therefore locally rejected. All 120 raw samples are
preserved in abba.json, along with exact cache/reset/timer conditions.

The observed slowdown cannot be attributed uniquely to local-memory footprint,
gather shuffles, repeated K/V reads, or GEMM efficiency from this experiment.
It establishes the complete-chain cost of this one unchanged mapping. No
additional tile, stage, warp, gather implementation, or timing round was tried;
there is no production change or E2E measurement. GPU was released immediately
after the single ABBA process exited. Invocation timestamps and full logs remain
in the result directory; the actual snapshot remains in ignored artifacts.

## Independent BM16 follow-up: CPU preparation

The M32 trial is complete. The subsequent bounded question changes only BM
from32 to16. It retains8 warps, one stage, QK BK32, PV BK64, full-row softmax,
both BF16 boundaries, and the same gather expression. There is no tuner or
parameter grid; the explicit --block-m option accepts16 or32 and defaults to32
so the prior invocation remains replayable. Existing M32 records are untouched.

At16 rows, the distributed FP32 score payload falls from128 to64 values/thread,
the BF16 probability payload falls from64 to32 KiB/CTA, and PV accumulator
payload falls from32 to16 values/thread. The QK staged operand payload becomes
65 KiB (64 KiB K plus1 KiB Q). These are payload arithmetic, not a prediction
of the compiler's register allocation or spill behavior. CTA count increases
from242 to484 and logical K/V reads double from239,878,144 to479,756,288 B/call.
This tests the net tradeoff; M32's slowdown is not attributed to spills alone.

Use the existing artifacts/rtx5090-pi05/backbone-fullrow-024-17.safetensors.
There is no new capture or model load. After a separate resource authorization,
compile only BM16 and save resources-m16.json/resources-m16.ptx plus its
invocation/log. Check both BF16 boundaries in the new PTX. Then wait for the
owner's decision before actual-input check and the single predefined ABBA:

    python -m lab.pi05.backbone_fullrow resources --block-m 16       --output "$RESULTS/resources-m16.json"
    python -m lab.pi05.backbone_fullrow check --block-m 16       --snapshot "$SNAPSHOT" --output "$RESULTS/check-m16.json"
    python -m lab.pi05.backbone_fullrow time --block-m 16       --snapshot "$SNAPSHOT" --output "$RESULTS/abba-m16.json"

All outputs, PTX, invocation records and logs use the -m16 suffix; they never overwrite M32 artifacts.
The actual-input check and timer reuse the identical17-layer snapshot, common
Q/out addresses, equal Q reset, shallow tolerance, warm working set, and
4 chains/graph with30 samples/leg. Failure stops this BM16 trial. There is no
BM8, warp, stage, BK, or gather sweep. CPU syntax checking passed; no new
kernel compilation or GPU work has run.

## BM16 resource and actual-input result: rejected

The authorized BM16-only compile completed with 206 registers/thread,
n_spills=0 (0 B/thread local footprint), 66,560 B shared (65 KiB), and 484 CTAs.
It used eight warps and one stage as prescribed. The two BF16 boundaries
survived: resources-m16.ptx533–596 rounds QK scores to BF16,598–661 widens them
before scale/mask;1286–1349 performs full-row normalization,1351–1382 rounds
probabilities to BF16x2, and the subsequent BF16 ldmatrix operands enter
PV mma.sync at3834. The selected original-numbered evidence is saved in
ptx-boundaries-m16.txt and ptx-evidence-m16.json.

After those checks, a separate explicit authorization allowed use of the same
024 snapshot. No model or capture was rerun. All 17 candidate outputs passed
with worst rel_rms5.217102491502758e-5, minimum cosine0.9999999986391401,
maximum absolute error0.03125, and finite output everywhere. Control outputs
remained exactly equal to the captured outputs.

The only BM16 ABBA retained the full chain, equal Q reset, shared Q/out
addresses, original cache policy, and 30 samples per leg:

| Leg | Median ms per 17 calls |
| --- | ---: |
| A1 control | 1.1596199870109558 |
| B1 BM16 | 1.8588799834251404 |
| B2 BM16 | 1.8592239618301392 |
| A2 control | 1.1904000043869019 |

BM16 was slower by0.6840419769 ms per17 calls (40.23776 us/call). Control
drift was0.0307800174 ms and candidate drift0.0003439784 ms; the closest
A/B medians remained0.6684799790 ms apart. It is clearly locally rejected.

The reduced measured registers and zero local footprint did not yield a net
advantage over this trial's control. This does not isolate the cost of doubled
K/V reads, gather, register pressure, or scheduling. M32 and M16 ran in separate
ABBA windows; their difference is not a simultaneous controlled comparison.
Both original M32 records and all120 BM16 raw samples remain. GPU was released
immediately after timing. There is no production change and no further BM,
warp, stage, BK, gather, or timing search.
