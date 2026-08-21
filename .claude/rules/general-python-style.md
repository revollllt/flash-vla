---
paths:
  - "**/*.py"
---

# General Code Style

Default conventions for new and modified Python code. Prefer these unless
there is a concrete reason not to; call out deviations in review.

- **Prefer stateless.** Favor pure functions over methods that mutate instance
  state; pass inputs in and return outputs out.
- **Prefer immutable.** Default to tuples, read-only values, and freshly-bound
  tensor fields. Mutate only when the buffer is explicitly part of a CUDA Graph
  or workspace contract.
- **Extract init-static values at construction.** If the inputs are frozen for
  the object's lifetime, compute the derived value once in `__init__` and give
  it a meaningful name.

  ```python
  # Good: the hot loop reads a value whose inputs cannot change.
  self.needs_cpu_seq_lens = self.backend == "cuda" and self.capture_enabled

  # Bad: repeat configuration logic for every action step.
  if self.backend == "cuda" and self.capture_enabled:
      ...
  ```

  If an input can change, do not cache it silently; recompute it or funnel the
  change through one explicit override point.
- **Functions stay small.** Keep functions roughly under 100 LOC. The main
  pipeline should read like pseudocode, with shape checks, allocation, and
  backend details in named helpers.

  ```python
  # Good
  def run_step(batch, *, backend):
      state = _prepare_state(batch)
      features = _encode_features(state)
      return backend.forward(features, state=state)

  # Bad: one 150-line function interleaves validation, allocation, kernels,
  # logging, and result post-processing.
  ```

- **Prefer keyword arguments.** For two or more same-typed or optional
  arguments, make the call site self-documenting:

  ```python
  # Good
  _launch_decode(x, block_m=block_m, block_n=block_n, stream=stream)

  # Bad
  _launch_decode(x, block_m, block_n, stream)
  ```

- **Pass what you need, not a god object.** Pass the specific read-only values a
  helper consumes instead of the whole pipeline or model object.
- **Keep ownership boundaries explicit.** Model contracts stay hardware-free;
  target-specific pipelines own buffers, fusion, and backend selection.
- **Typed public contracts.** Document tensor shape, dtype, device, mutation,
  and CUDA Graph capture constraints for public functions. Do not add per-call
  validation to a captured hot path when the invariant can be checked during
  construction or compilation.
