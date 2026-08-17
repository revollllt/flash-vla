# Architecture

This repository builds fixed-workload VLA inference targets for a specific GPU,
model revision, shape profile, and precision policy. Peak performance takes
priority over a universal operator abstraction.

## Target ownership

The atomic production unit is a hardware/model target:

```text
Target = device profile x model revision x shape profile x precision policy
```

`hardware/nvidia/h100/pi0` therefore owns the Pi0 execution plan for H100,
including its pipeline, static buffer plan, call-site wrappers, kernel source,
fusion boundaries, and selected tuning. Workload-specific kernels are not
placed in a global architecture-only kernel directory.

The target is implementation-strategy agnostic. Each call site resolves to a
backend through `ops.op_table(plan=...)`; TileLang is the current main line — a
performance/effort trade-off for agile development — and hand-written CUDA is
planned for the remaining performance. A single pipeline may mix backends per
call site without the pipeline knowing which is which.

## Repository layout

```text
src/flash_vla/
  models/                         hardware-independent model contracts
    pi0/                          checkpoint schema and weight helpers
  runtime/
    cuda/                         graph-safe runtime mechanisms (ScratchPool)
  hardware/
    nvidia/
      h100/
        pi0/                      one deployable target
          engine.py               weights, capture, and public forward
          pipeline.py             in-place execution plan
          buffers.py              target-specific static buffer plan
          ops.py                  per-call-site backend binding
          backends/               backend registry (one module per strategy)
            tilelang/             TileLang backend (current main line)
              wrappers.py          call-site configs and launch wrappers
              fused_wrappers.py    fused call-site alternatives
              kernels/             H100/Pi0 workload-specific TileLang kernels

eval/                             numerical and policy-quality evaluation
benchmarks/                       performance measurement and profiling
```

## Dependency direction

```text
eval / benchmarks -> public engine API
hardware target   -> models + runtime + target-local backends
models            -> no hardware target
runtime           -> no model or hardware target
backend kernels   -> their backend and its own toolchain (TileLang / CUDA)
```

A backend module registers `ALL_WRAPPERS` / `FUSED_WRAPPERS` in
`hardware/.../pi0/backends/`; the target pipeline consumes whatever the op table
hands it. Production targets must not import official baselines or evaluation
suites. One target must not import another target's private kernels. If two
targets eventually use the same implementation without model/device branches,
extract the proven common part then; do not generalize a kernel in anticipation.

## Specialization rules

- Resolve operation bindings before CUDA Graph capture; never dispatch by model
  or device in the replay path.
- Pass destinations and workspaces explicitly. A captured execution must not
  allocate device memory.
- Keep fusion boundaries target-local because they change the pipeline and
  buffer lifetimes, not just one operator implementation.
- Keep tuning results with the target and call site that produced them.
- Add a separate execution plan only when a shape profile changes topology or
  fusion, rather than merely changing a compile-time constant.
- Treat PDL as a target pipeline decision: kernels expose the required device
  control points, while the target owns the dependent launch chain.

## Validation

Correctness has two independent gates:

1. Numerical equivalence against an official baseline, from individual stages
   through the final action tensor.
2. Policy-quality evaluation such as LIBERO task success.

Performance benchmarks are separate from both gates and must measure the real
captured shape/profile. A faster result cannot override a correctness failure.
