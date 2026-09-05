# Target Composition and Dependency Rules

Status: implemented for the layout and rules marked as such; items marked
planned follow [`10-runtime.md`](10-runtime.md).

## Ownership map

```text
src/flash_vla/
  models/<model>/          hardware-independent model contract: constants,
                           checkpoint schema, weight folding, torch reference,
                           tokenizer
  runtime/                 identity, plan binding, the engine protocol
  runtime/cuda/            graph-safe mechanisms: arena and scratch pool,
                           segments and their lifecycle, in-graph timing
  tuning/                  backend-agnostic config sweeps: candidate sets and
                           the sweep loop; no model, backend or device knowledge
  hardware/nvidia/
    cuda/tile/sm90/        shared SM90 tile primitives with their own parity suite
    h100/spec.py           static device spec
    h100/<model>/          one Target
      engine.py            construction and the online path
      pipeline.py          orchestration: call sites only, in place on buffers;
                           call-site cost declarations (planned)
      buffers.py           the buffer plan, declared as data
      ops.py               op-table construction from a validated plan
      backends/<name>/     one module per implementation strategy: wrappers,
                           fused overlays, kernels, tuning adapter
eval/
  acceptance.py            the acceptance registry: framework defaults and
                           per-Target entries, device-free (planned)
  baselines/               adapters to official implementations
  correctness/<model>/     parity scripts; the shared metrics module (planned)
  tasks/                   policy-quality suites (out of scope this phase)
benchmarks/                latency and profiling harnesses, plan registry,
                           the floor model (planned)
```

## Dependency direction

```text
eval / benchmarks  -> engine protocol + runtime + models; never backend internals
hardware target    -> models + runtime + its own backends
models             -> nothing hardware-specific
runtime            -> no model, backend or hardware target
tuning             -> runtime only
backend kernels    -> their backend and toolchain; CUDA kernels -> tile primitives -> cutlass
backend adapters   -> tuning + their own device spec
```

Production code under `src/` must not import `eval` or `benchmarks`. One
Target must not import another Target's private kernels. A backend registers
its wrappers with its Target; the pipeline consumes whatever the op table hands
it and does not know which backend it is.

## Model contract versus runtime shapes

The model contract is one entry per upstream tensor up to lossless,
schedule-independent relayout: transposes, projection concatenation, RoPE
channel permutation, absorbing a plain per-channel norm scale into the GEMM
that consumes it. What a Target loads may differ: constants that are pure
functions of the fixed inference schedule are folded at load. The dividing
line is "does it depend on the inference schedule"; everything that does stays
out of the model contract and inside the Target.

## Specialization rules

- Resolve call-site bindings before capture; never dispatch by model or device
  on the replay path.
- Pass destinations and workspaces explicitly. A captured segment must not
  allocate device memory.
- Declare the buffer plan as data; the runtime materializes it. The Target
  owns padding, masking and aliasing, and documents any pad region a kernel
  writes.
- Keep fusion boundaries Target-local: a fusion changes the pipeline and the
  buffer lifetimes, not only one call site.
- Keep tuning results with the Target and call site that produced them. The
  sweep loop is workload-independent and lives in `tuning/`; the adapter that
  turns a candidate into a compiled kernel is backend-private.
- A device capability is not a tuning axis. The spec states what the hardware
  can do; the backend adapter decides what that implies for its own axes. The
  spec never names a backend concept and `tuning/` never names a hardware one.
- Derive a tuning space from the spec rather than filtering a fixed list;
  shared memory per block is usually the binding limit.
- Add a separate execution plan only when a shape profile changes topology or
  fusion.
- Treat PDL as a Target pipeline decision: kernels expose the control points,
  the Target owns the chain.
- Backends declare their route constraints; the runtime validates a plan
  against them at engine construction.

## Validation ownership

- Correctness has two independent tensor-level oracles (in-engine reference
  route, official baseline) and one policy-quality level; see
  [`32-correctness-evaluation.md`](32-correctness-evaluation.md).
- Latency is measured on the real captured workload only; see
  [`31-latency-evaluation.md`](31-latency-evaluation.md).
- A faster result never overrides a correctness failure.
- The parity script that owns a rounding contract is the authority when a
  number is in dispute.
