---
paths:
  - "**/*.cu"
  - "**/*.cuh"
  - "**/*.cpp"
  - "**/*.hpp"
---

# CUDA and C++ Style

These are defaults for new CUDA/C++ and FFI-facing code. Match the existing
backend when changing a narrow area, and record a measured reason for a local
exception.

- **Use fixed-width integers at boundaries.** Make narrowing explicit.

  ```cpp
  // Good
  int32_t tiles = static_cast<int32_t>(num_tiles);

  // Bad: width and signedness depend on the host/compiler.
  int tiles = num_tiles;
  ```

- **Express the aliasing contract.** Read-only pointers are `const`; add
  `__restrict__` only when the caller guarantees non-aliasing.

  ```cpp
  // Good
  void load_tile(const half* __restrict__ input, half* __restrict__ output);

  // Bad: mutable input and an unspoken aliasing assumption.
  void load_tile(half* input, half* output);
  ```

- **Check errors at the cheapest useful boundary.** A host launch wrapper should
  report the kernel name and launch parameters; a device helper should not add
  a per-element error path.
- **Keep layout decisions traceable.** Tile sizes, stages, warp roles, barriers,
  register/shared-memory budgets, and tail policy must map to a TileDataflow
  section, backend reference, or measured configuration.
- **State synchronization ownership next to phase changes.** Do not hide a
  barrier or producer/consumer hand-off behind an unexplained helper.
- **Protect the register budget.** Recompute a cheap value when that avoids a
  spill, and record the measured register count when it is a constraint.
- **Keep source comments ASCII-only.** Use `->`, `<=`, and `--`; reserve Unicode
  mathematical notation for Markdown specifications.
