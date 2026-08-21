---
paths:
  - "**/*.py"
  - "**/*.cu"
  - "**/*.cuh"
  - "**/*.cpp"
  - "**/*.hpp"
---

# Comments and Docstrings

Comments explain a decision that is not recoverable from the code. They should
name the invariant, constraint, or evidence and stay next to the code it
protects. Do not add a comment merely because a line is non-empty.

- **Explain why, not what.**

  ```python
  # Good: correctness here is tiling-dependent -- some tilings, BLOCK_M=32 among
  # them, produce garbage rather than an error. Re-validate numerically after
  # any re-tune.
  BLOCK_N = 32

  # Bad: Set BLOCK_N to 32.
  BLOCK_N = 32
  ```

- **Make synchronization comments precise.** Name the producer/consumer and
  the level of synchronization. “Synchronize here” is not sufficient.

  ```cpp
  // Good: wait for the previous stage's global stores before this CTA reads
  // the double-buffered tile; this is a phase dependency, not a CTA barrier.
  pdl_wait(previous_stage);

  // Bad
  // Synchronize.
  pdl_wait(previous_stage);
  ```

- **Keep kernel comments at the implemented dataflow level.** A useful comment
  identifies the grid, stage, warp role, instruction, or epilogue and points to
  the relevant TileDataflow section when the choice is non-obvious.
- **Document public contracts.** State purpose, input/output shape and dtype,
  device, mutation/aliasing, and whether the function is safe during CUDA Graph
  capture. A docstring such as `"Run the pipeline."` is not a contract.
- **Do not hide magic numbers.** Either give a constant a meaningful name and
  cite its spec/benchmark source, or explain why it is intentionally local.
  Update the comment when that source changes.
- **Comments are not evidence of speed.** If a comment claims a performance
  effect, include the benchmark mode or profiler artifact that established it;
  otherwise phrase it as a hypothesis.
- **Use plain ASCII in C++/CUDA comments.** Markdown specs may use Unicode;
  source comments should use `->`, `<=`, and `--`.
