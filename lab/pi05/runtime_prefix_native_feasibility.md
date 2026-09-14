# Runtime prefix buckets: cfg0 native feasibility

This is a CPU source review, not an implementation or a performance result.
Native source was inspected at main `2e7deee`; production code is unchanged.

## Workload evidence and scope

The backbone worker inspected two independent actual seed-42 safetensors
recordings from `bf2a9ba` and `6f8aec5`. Each recording contains nine masks;
all have 895 valid prefix entries, 127 prompt tokens, and negative `mask[896]`.
Thus the proposed 896-row bucket is exercised by the current timing workload.
The seed-0 oracle has 903 valid prefix entries and exercises the 968-row bucket.
These are reported recording observations, not inferred from the tokenizer.

External tensor shapes remain 968 rows. The device mask selects one of two
static GEMM parameter sets, M896 or M968, on each replay. This selection relies
on the existing contiguous valid-prefix layout; checking one mask entry does
not in general prove an arbitrary mask has no valid entries after that point.
The explicit graph mask wiring is a separate Target review.

## Reusing cfg0

The existing cfg0 type is in
`src/flash_vla/hardware/nvidia/rtx5090/pi05/backends/cutlass_backbone.cu:16`.
Its vendor kernel exposes `GemmKernel::invoke(Params const&, SharedStorage&)`
in `third_party/cutlass/include/cutlass/gemm/kernel/gemm_universal_streamk.h:1122`.
It constructs the existing operator and runs its unchanged mainloop/epilogue.

A small Target-local adapter can inherit this kernel's types, extend
`Arguments` and `Params` with the device mask and bucket selection, and perform
a CTA-uniform early return before calling the original `invoke`.
All CTAs of a launch must agree, including Stream-K reduction CTAs.
The mask must be produced before both launches on the same dependency chain and
remain unchanged during them. No vendor mainloop or epilogue copy is needed.

Use `GemmUniversalBase<adapter>` for the adapted entry. Its device initialization
configures `Kernel2<adapter>` dynamic shared memory and queries that exact entry's
occupancy (`gemm_universal_base.h:129`), which also determines Stream-K Params.
Simply reusing an original plan's Params with a new global entry would miss
this initialization and could use the wrong occupancy/workspace mapping.
The existing `params_` is protected, not a public plan-handle accessor.

Both entries must live in the existing native library. The original control
continues through the original `Gemm` and `Kernel2<GemmKernel>`; the candidate
uses a distinct adapter type, hence distinct template initialization state.
They reuse the same cfg0 mainloop and epilogue source, but are different global
kernel symbols. This avoids another DSO with the same GNU-unique TLS state.
Actual register/shared usage and the adapter's occupancy remain unmeasured.

Native changes would be limited to this adapter and its workspace/plan/run/
destroy boundary. Python would retain both static plans, tensor references,
and workspaces in one wrapper instance; both plans must be initialized during
warmup before capture. No CPU mask read or graph recapture is required.
Exact graph signatures and ABI details await the explicit-mask design.

Both fixed launches remain in the graph. An unselected launch still schedules
CTAs with its compiled resource requirements before returning; it does not
remove launch overhead. The selected launch executes the entire original
Stream-K mapping. No latency gain is established by this feasibility review.

## Workspace and finite-tail requirements

Keep the two buckets' workspaces disjoint and instance-owned. Vendor Params
places partials first, then barrier flags at an offset depending on the
Stream-K mapping (`gemm_universal_streamk.h:294-417`). A bucket's partial writes
could otherwise overwrite the other bucket's barrier location. Separate
scratch roles make this independent of whether the allocator happens to
separate their shapes. Successful active invocations reset barrier flags
through `wait_eq_reset` (lines 809 and 885); an inactive entry must touch none
of the workspace or output.

M896 leaves rows 896:968 unchanged. If following full-shape pointwise kernels
read those gate/up buffers, the tail must already contain finite values.
The existing runner Scratch allocates with zeros
(`src/flash_vla/runtime/runner.py:41`), which provides finite initial scratch;
persistent reuse and any external output alias must also be checked across
bucket transitions. Skipping writes alone does not sanitize a stale tail.
This review does not propose an unconditional full-buffer clear.

Backbone down uses the existing FP32 linear epilogue with beta=1 and C=D.
The selected bucket must read the incoming residual and update active rows
exactly once. The inactive bucket must not modify C/D; clearing the whole
residual buffer to initialize padding would destroy that behavior.
The short bucket intentionally leaves the masked tail unchanged. Full-shape
consumers and later valid rows on a long replay must receive freshly computed
finite inputs before their use. Gate/up retain their BF16 stores before GELU;
down retains its existing accumulation/add/store semantics.

## Required validation if implemented

Compilation must first confirm the adapter entry's actual resources, launch
attributes, mapping and workspace sizes. Numerical validation must exercise
short → long → short on the same captured graph and persistent buffers, with
both routes compared against the ordinary control on the same inputs.
This specifically covers stale-tail reads, residual aliasing, and workspace
reuse when the device mask changes. The seed-0 oracle alone covers only the
long bucket and is insufficient. Changing M can change the Stream-K reduction
mapping, so unchanged epilogue source does not imply bitwise parity.
Only a passing candidate should enter the existing bounded timing workflow.
