# Pi0.5 prefix padding and explicit runtime row selection

CPU-only design record on source `2e7deeebed244ddca7574aa0d7a4794c3484fa1e`.
No model, Torch import, CUDA context, compiler, JIT or GPU was used for this
investigation. This revision changes documentation and compact evidence only.

The padding is semantically removable, but the deployed dense kernels do not
already remove it. Existing Target graph extensions can pass the current device
mask explicitly without changing the shared runner, host slot, external inputs,
fixed KV allocation, or initial-capture reuse policy. The existing Stream-K plan
cannot take a changing device M directly. A fixed 896/968 selection is a smaller
hypothesis than replacing its scheduler with grouped GEMM; native feasibility is
recorded separately by the native worker. No implementation is proposed from
tile counts alone.

## Actual input screen

The default task is `pick up the plate and put it in the sink`
(`src/flash_vla/inference.py:28`). Latency calls
`build(..., seed=seed)` then `engine.sample_inputs(seed)`
(`benchmarks/latency.py:240`). The latter draws images, state, noise in that order
from one generator **on the runner device**, then copies state to pinned CPU
memory (`src/flash_vla/runtime/vla.py:149`; Pi05 `INPUTS` at
`src/flash_vla/hardware/nvidia/h100/pi05/target.py:87`).
A CPU generator with seed 42 would not reconstruct the measured state.

Instead, two existing snapshots provide the actual runtime mask. Both were
recorded by `lab/sm120/pi05_attention_qk_triton_confirm.py:19`:
build at seed 42, `engine.sample_inputs(42)`, stage those inputs, execute the
actual host/program, capture Q/K/V/mask before attention. The default task is not
overridden. The softmax experiment imports that same prepare function.

| Evidence | Source at capture | Cases | Prompt tokens | Valid prefix rows |
| --- | --- | ---: | ---: | ---: |
| QK snapshot | bf2a9ba86d5e25d444583b234fe3dd3c346ed998 | 9 | 127 | 895 |
| Softmax128 snapshot | 6f8aec5125349b7e820d857d3a25c188de70af18 | 9 | 127 | 895 |
| Existing official oracle, seed 0 | official-eager.json / 021 official report | one fixture | 135 | 903 |

All 18 saved masks have zero at rows [0,895), MASK_NEG at [895,968),
and zero at [968,1018). Row 896 is masked. Exact file paths and compact
per-case raw-value evidence are in
[`padding_runtime_rows.json`](../../results/rtx5090-pi05/gpt6-padding-cpu/padding_runtime_rows.json).
The snapshots were read using only Python `json` and `struct`; their BF16 mask
bytes were not converted by a GPU operation.

Thus the current latency fixture would choose M896 and still compute one masked
row (895). The official fixture chooses M968. Neither length is a model
invariant. Different state values change the number of state digit tokens;
`set_task` changes the task prefix, and tokenization truncates at 200.

## What can change at each replay

Source paths in this section are relative to `src/flash_vla/`.

- `models/pi05/tokenize.py:176` encodes the installed task plus current state.
  `_finish:220` writes valid tokens first, then zero token IDs and a false mask.
  Validity comes from the mask, not from token value zero.
- `hardware/nvidia/h100/pi05/prefix.py:117` calls that tokenizer every host slot,
  sets host `n_valid = 768 + n_tokens`, and selects four staging buffers:
  prompt IDs, embedding scale, additive key mask, and expert RoPE.
  `copy_into:139` updates their already allocated device addresses.
  There is currently no device n_valid scalar.
- `pipeline.py:135` puts this host slot after vision starts and before the
  backbone replay. The next forward can have different task/state/token values
  and length. During one ten-step expert replay the prefix mask and prefix KV
  remain fixed; suffix KV changes by step/layer. Replaying a segment alone uses
  the buffers most recently staged by the host.
- `pipeline.py:90` initially makes the logical mask all zero for warmup.
  Prompt IDs and scales start at zero. The first capture therefore cannot be
  used to freeze a semantic length. A device-read mask observes later updates.
- `pipeline.py:143` uses static prefix positions 0..967. This is correct for
  every valid prefix because language validity is contiguous.
  `prefix.py:106` sets expert logical positions to n_valid + [0,50).
- `pipeline.py:145` allocates KV as [18,1024,256], exposes [18,1018,256],
  prefix [0,968), suffix [968,1018). **The suffix physical offset stays 968**,
  even though its logical RoPE positions begin at n_valid. No compaction is
  needed or allowed in the proposed internal selection.

All three image views are present in the current external INPUTS; there is no
runtime image-valid mask. This analysis concerns only the 200 language slots.

## Why padded query results may differ

The embedding writes zero on padding every forward. RMSNorm, the projections,
GELU/product and residual operations are row-local. Attention is the only
cross-row dependency, and every later backbone/expert attention masks those
prefix positions as keys.

The deployed backbone attention is the full dense torch chain:
`hardware/nvidia/h100/pi05/backends/tilelang/kernels/attention.py:52`.
It computes all 7744 query-head rows against all 968 keys, then softmax and PV.
The key mask does not skip query work. Its own source comment at line 16
already explains that padded query outputs differ from upstream and cannot
propagate to valid rows. Pad rows generally become nonzero after attention;
zero-valued x is not a valid detector. The final backbone layer already omits
attention/FFN entirely, but the preceding layers do not omit padding.

`eval/pi05/parity.py:217` compares valid prefix KV only and line 222 requires
all padded KV to remain finite. Actions remain fully compared. A skipped row
may be finite zero, but it may not be an uninitialized/NaN value: dense later
loads can turn 0 * NaN into NaN. If a subsequent task/state makes a previously
padded row valid, that row must be recomputed on that replay.

For an FFN-only bucket candidate, keep QKV and attention dense initially. Ensure
the up/GELU output is finite on every row the unchanged down/control path could
read. Do not depend on scratch contents left by warmup or the previous request.
In particular, initial zero scratch is insufficient: after a long request,
short-M GEMM leaves the last 72 rows of gate/up untouched, and repeatedly running
the original full-size in-place GELU/product could repeatedly multiply those
stale rows and eventually overflow. The smallest tail policy is for the
Target-local GELU path to read the same mask selection, directly write zero to
[896,968) on the short bucket, and never load those stale gate/up values. Keep
RMSNorm full-size. Short-bucket down leaves the already finite residual from
attention untouched on those rows; there is no need to clear the whole residual.
The long bucket rewrites all gate/up rows before full-size GELU. Native changes
wait for the pure-GEMM static upper-bound probe.

A later QKV bucket candidate would also need a finish path that writes finite
padding before reading any uncomputed projected value.

## Smallest explicit Target graph path

`runtime/ops.py:161` gives QKV seven arguments; FFN/up at line 171 and down at
174 also have no mask. Attention already has a mask argument at 165.
A backend factory receives an allocator and optional asset paths, not the
runner's device buffers (`runtime/registry.py:73`). Capturing mask through
shared scratch would hide a real data dependency and is unnecessary.

The existing extension mechanism is sufficient:

1. Add two **new** Target extension OpSpecs for masked FFN up and masked down,
   copying each standard spec's parameter/output/weight/inout/aux/cost declarations
   and appending a read-only `mask` parameter. Do not redefine the standard
   names: `Vocabulary.__init__` at `runtime/ops.py:208` rejects shadowing.
2. Override only `Pi05RTX5090.build`: call the inherited build, obtain
   `g.buf("mask_bias")[:shape["prefix_len"]]`, and replace the selected existing
   FFN Nodes with the extension call-site name and original args plus this
   BufRef. Preserve node index, stage and every activation/weight/output ref.
   Node is an immutable dataclass but `g.nodes` is an ordered mutable list;
   `dataclasses.replace` makes the small rewrite explicit. No inserted node,
   new host slot or copied shared pipeline is needed for a two-bucket test.
3. Register the new backend's `OPS` through the existing Target Registry.
   `VLA.graph:139` creates the extended vocabulary before Target.build, then
   performs its existing graph check. `Graph.reads_writes:303` now sees the
   mask as an actual input.
4. Each candidate wrapper receives that same fixed mask address and passes it
   to its native entry. It launches both statically prepared buckets every
   eager/capture invocation. Device entry selection reads mask[896] on every
   replay: negative selects M896; zero selects M968. The short bucket is legal
   because contiguous validity makes mask[896] negative iff n_valid <= 896.
   An explicit n_valid buffer, device reduction, additional H2D transfer, or
   cross-op metadata channel is unnecessary for these two fixed buckets.
5. Keep standard BF16 materialization, pointwise order, and fixed output
   allocation. A wrapper owns both initialized plan handles; scratch owns any
   persistent device workspace. Prepare both during warmup even if only the
   long path performs work on initial zero masks. No allocation or Python
   length branch can be deferred until a different mask reaches capture/replay.

This changes internal call-site names, so control routing is part of the
minimum change, not optional bookkeeping. `runtime/binding.py:39` accepts plan
keys known to any backend even when absent from the current graph; naively
renaming nodes can silently ignore an old saved plan's standard FFN keys.
A Target-local `select_plan` must map those two old keys to the new call sites
before Registry resolution. The unchanged torch and current cutlass-backbone
controls need thin aliases accepting and ignoring mask, so reference and
saved 021 control plans preserve their original dense work. The candidate uses
the mask. No shared binding/registry policy needs changing; other unsupported
legacy backend aliases should fail the existing provided-name check explicitly.

Minimum prospective files for this **FFN-only** path:

| Target-local file | Necessary role |
| --- | --- |
| `target.py` | Narrow build rewrite and explicit two-key plan translation |
| `backends/__init__.py` | Register candidate extension specs/factory |
| New `backends/bucketed_backbone.py` | Two concrete OpSpecs and candidate wrappers, owned plans |
| `backends/torch_ops.py` | Two thin reference aliases that discard mask |
| `backends/cutlass_backbone.py` | Dense control aliases; existing loader may bind added ABI |
| `backends/cutlass_backbone.cu` | Native two-static-plan entry/ABI, subject to native feasibility |
| Pointwise native source, only if required | Explicit finite writes for skipped up/output tail |

These are design boundaries, not a request to implement all sites. QKV would
use its own analogous extension and dense adapter only if separately justified.
No shared runtime, model tokenizer, H100 pipeline, inference factory, precision,
external INPUTS, fixed shapes, or KV layout changes are required.

The existing lifecycle remains intact:
`runtime/runner.py:182` resolves all refs once;
`runtime/cuda/program.py:57` warms and then freezes scratch;
line 64 captures each segment once; line 71 replays it. Python `.item()`,
host length-dependent slicing, recapture, and per-request graph switching are
not the proposed mechanism.

## Existing CUTLASS limits and the alternative not selected

Current cfg0 uses `GemmUniversal` with `ThreadblockSwizzleStreamK`.
In the vendored `include/cutlass/gemm/kernel/gemm_universal_streamk.h:126`,
problem_size is a by-value argument. The Params constructor builds the
Stream-K mapping from it. The swizzle constructor
(`threadblock/threadblock_swizzle_streamk.h:402`) computes tiled shape,
tile count and decomposition on the host. Params::update changes pointers,
strides and epilogue, not this mapping. The captured kernel receives Params by
value. Thus **existing _Plan plus a device length pointer is not enough**.

Vendor `GemmGrouped` has a device-only visitor with a device
`GemmCoord *problem_sizes` (`kernel/gemm_grouped.h:136`;
`kernel/grouped_problem_visitor.h:205`). Its next_tile reads those sizes,
and `device/base_grouped.h:350` initializes without host precomputation for
that mode; workspace size is zero and launch block count can remain fixed.
That could support a mutable M with stable graph addresses, but it is a
different scheduler/kernel, not reuse of current Stream-K cfg0. Full-M control
performance and numerics would first need measurement. It is more scope than
a fixed two-static-mapping experiment and is not selected here.

The parent's two-bucket proposal keeps both existing mappings static and makes
the entire inactive launch return at entry. This must be checked against native
Params accessibility, barriers, workspace lifecycle and finite tails by the
native analysis. It does not mean skipping arbitrary Stream-K block IDs:
partial producer/consumer skips could break its reduction/barrier protocol.

## Work bounds, not predicted latency

Let t be the number of language tokens. There are 968 fixed rows, 768+t valid
rows and 200-t padding rows. Exact query-only pruning has a logical work fraction
(200-t)/968, with keys still occupying 968 physical positions.

| t | Valid rows | Logical padded-row fraction | M128 row tiles at that M |
| ---: | ---: | ---: | ---: |
| 0 | 768 | 20.6612% | 6 |
| 64 | 832 | 14.0496% | 7 |
| 127 (latency snapshot) | 895 | 7.5413% | 7 |
| 128 | 896 | 7.4380% | 7 |
| 135 (oracle) | 903 | 6.7149% | 8 |
| 160 | 928 | 4.1322% | 8 |
| 200 | 968 | 0% | 8 |

This table spans the supported pad length; it is not a measured task/state
distribution. Fixed 896/968 buckets remove one of eight M128 row tiles only for
t <= 128, retain all eight for t >= 129, and do not take the second possible
reduction at t=0. For the measured latency state, 72 rows are skipped rather
than all 73 padding rows. Smaller grid/Stream-K decomposition can change
arithmetic association and efficiency; an extra inactive launch and tail
handling have costs. None of these percentages predicts end-to-end speed.

Attention-only query skipping can read the existing mask with no OpSpec
extension, at the same dynamic logical-row fraction (0–20.66%, 7.54% on the
recorded latency input). It would need a backend that actually omits padded
queries; the present torch mask alone does not do so. Separate numeric/kernel
resource risks make that a distinct candidate, not a free saving to add.

The information-gain check for a future bucket probe is first whether the exact
current seed42 full chain gains more than same-condition drift, followed by
same-captured-graph length transitions across <=896 and >896 with valid output
comparisons and finite tail checks. The existing seed0 official fixture only
exercises the long bucket and cannot validate short-bucket numerics alone.
No implementation or measurement is included in this CPU record.

## Python candidate integration

The optional `bucketed-backbone` backend now implements
`llm_backbone_norm_gated_ffn_masked` and
`llm_backbone_ffn_down_residual_masked`. Target.build appends the existing mask
view to those two call sites; Target.select_plan translates their old standard
keys, and an explicit masked key takes precedence. The shipped selection stays
on dense `cutlass-backbone` aliases; reference uses dense `torch` aliases.
Select the candidate by overriding only those two backend values.

The Python binding expects the native ABI from `479a2d8`, in the existing
`cutlass_backbone.so` and `fused_backbone.so`. Each pointer set owns both static
native handles and separate scratch roles for M896/M968. New workspace queries
use the bucket entry's occupancy and expose negative error returns. The mask
is passed at each native run and GELU launch; no runtime metadata goes through
scratch. The original dense wrappers and their arithmetic remain the control.

CPU validation, with CUDA hidden and compiler environment unset:

- `python -m pytest -q tests/test_pi05_bucketed_routes.py tests/test_binding.py`:
  9 passed. Includes old complete-plan loading, two-site candidate selection,
  explicit masked override precedence, reference routing, and mask/layout checks.
- `python -m tests.targets --target rtx5090/pi05`: 8/8 declaration checks and
  1/1 route checks passed.
- The saved 022 control plan (the retained 021 implementation) binds to the
  same complete backend map after only the two call-site names are translated.

These checks import Torch on CPU but do not initialize CUDA or load models.
Native compilation, numerical transitions and complete-chain timing remain
with the serial integration task; no candidate speed is claimed here.
