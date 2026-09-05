# Static Inference Runtime

Status: implemented. The arena, scratch pool, segment list with host slots,
shared lifecycle, in-graph timing, plan validation against backend-declared
route constraints, the identity and the engine protocol live under
`src/flash_vla/runtime/`; both Targets' engines construct through them. No
generic harness consumes the protocol yet (see the evaluation documents).

## What the runtime is

The runtime is the set of mechanisms that are invariant across Targets, and
nothing more. It has no model, backend or device dependency: it does not
branch on any of them, it does not import any of them, and it cannot be
extended by adding a model or a device to it. The boundary rule is the only
design rule: **what is invariant across Targets goes into the runtime; all
else goes into the Target.**

The runtime is not a compiler. It never infers buffer lifetimes, aliasing or
padding, never reorders or fuses call sites, and never tunes anything. Those
are Target decisions, made by the agent, recorded in the Target.

## Mechanisms

### Static arena

The Target declares its buffer plan as data: for each buffer a name, shape,
dtype, initialization (uninitialized, zero, or a value), padding of the leading
extent, and any view or alias relation to another buffer. The runtime
materializes the plan once at engine construction, at fixed addresses that
never move for the engine's lifetime, and hands the Target the named buffers.

The Target's declaration is authoritative. The pipeline writes into these
buffers in place through explicit destination parameters; the runtime does not
know which call site touches which buffer.

### ScratchPool

Temporaries that a call site cannot avoid come from a pool keyed by role, shape,
dtype and device. Each key is allocated once and reused. After warmup the pool
is frozen; a request for a key warmup did not cover raises instead of
allocating during capture.

### Segments and host slots

A Target declares an ordered list of graph segments and the named host slots
between them. A segment is split off only for one of two reasons: a data
dependency on host work that must run between two segments (Pi0.5 tokenizes
the state on the host after vision starts), or a measurement need (a per-stage
latency split). Segments share the static addresses; the topology is fixed at
capture. Every segment is replayable alone, which is what makes per-stage
timing and stage-level oracle injection possible. A Target with no host work
and no measurement split declares one segment (Pi0 today).

### Lifecycle

```text
bind -> allocate -> load weights -> warmup -> freeze -> capture -> replay
```

- **bind** resolves the op table from the plan before anything is allocated.
- **warmup** runs every segment enough times to compile every kernel and fill
  the pool with every scratch key.
- **freeze** forbids further allocation.
- **capture** records each segment once.
- **replay** is the online path: stage inputs, run host slots, replay
  segments in order, return output views.

Invariants that hold from freeze onward: no device allocation; no dispatch by
model, device or backend; no host synchronization other than the declared host
slots; every kernel writes through a destination parameter.

### Binding

A backend is a flat registry of call-site wrappers with identical signatures,
stateless or built by a factory whose state follows the engine's lifetime.
The op table is built once per engine from the plan. Backends declare route
constraints (groups of call sites that must resolve to the same backend because
they share a buffer contract); binding validates a plan against every
backend's constraints and rejects violations at construction, not at the first
replay. There is no global dispatch state, because a benchmark routinely holds
two engines with different plans alive at once.

### Timing

Two mechanisms live here because the backend autotuner needs them and a
production package must not import a benchmark harness: capture of a callable
into a graph, and amortized in-graph timing of one kernel against cold inputs.
Benchmark policy (which statistic, how many reps, what gates) does not live
here; see [`31-latency-evaluation.md`](31-latency-evaluation.md).

### Inputs and outputs

Inputs are staged by copy into their static addresses. Outputs are views of
static buffers, valid until the next forward. A Target may use one buffer as
both input and output (the diffusion noise becomes the action chunk); the
buffer plan says so.

## The engine protocol

Generic harnesses (latency, correctness, promotion) are written against one
protocol that every Target's engine satisfies. The protocol is the runtime's
public face and the only thing the harnesses may depend on:

- **construction** from a Target identity, a shape profile, a plan and a
  checkpoint;
- **identity**: the block stamped into every report;
- **sample inputs**: seeded inputs at the Target's shapes, for random-weight
  measurement and comparison;
- **forward**: inputs in, output views out;
- **segments**: the ordered names, replay of one segment, and the host slots;
- **stage buffers**: named access to the buffers at segment boundaries, so a
  correctness harness can inject an oracle's stage output and read the
  Target's;
- **plan**: the resolved route.

Everything model-specific that a harness needs (which buffers form the KV
cache, which rows are valid) is exposed through these names by the Target,
never hard-coded in the harness.

## Graph facts the design relies on

- Graph replay removes host launch scheduling; it does not remove GPU grid
  ramp. Launch count, PDL overlap and fusion cost remain first-class design
  variables of the Target.
- PDL is a Target pipeline decision. Kernels expose the control points; the
  Target owns the dependent-launch chain and which boundaries overlap.
- Nothing inside a captured segment may synchronize with the host.
