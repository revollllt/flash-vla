# Harness Guide

A **profiling harness** is whatever process launches the kernel you want to profile, with realistic inputs, built with flags ncu can consume (`-lineinfo` for source-line attribution). Prefer an existing driver; build a standalone harness when it answers the question more directly.

---

## Option A (default here): the engine's own drivers

This repo is fixed-workload: the Target's production shapes are the only ones worth profiling, and the engine already has drivers that launch every kernel on them. Profile *through* them and let ncu's filter pick the launch:

| Driver | Launches | Use for |
|---|---|---|
| `python -m eval.correctness --target h100/pi05 --plan <plan> [--steps N --layers N]` | the full plan once, reference vs candidate | any shipped or `lab/` kernel at production shape; `--layers 1` keeps the job short |
| `python -m benchmarks kernels --target h100/pi05 --plan <plan> --site <site> --timer events --reps 1` | one call site, warmup + repeats | isolating one kernel; use `--launch-skip` to reach steady state |
| A compiled CUDA template with its built-in harness | one mechanism | a study outside the pipeline |

Getting `-lineinfo` in: the CUDA backends take extra nvcc flags from an environment hook — `ATTN_NVCC_DEFINES`, `FFN_NVCC_DEFINES`, `ENC_ATTN_NVCC_DEFINES` (space-separated). The build cache is hash-keyed on the flag string, so an instrumented `.so` compiles into its own cache entry and never displaces the production build. Export the hook in the process environment before the engine builds. TileLang kernels: pass `-lineinfo` through the TileLang build options of the plan, or profile without source lines (hotspots still resolve to SASS).

Exercise an unfamiliar driver once *without* NCU before adding the profiler: ncu's errors are far less descriptive than the runtime's.

---

## Option B: a standalone harness

Build a small CUDA executable that launches the kernel directly when:

- the kernel is not (yet) in the pipeline — a candidate, an ablation, a mechanism probe;
- the kernel lives inside a JIT/template build you cannot instrument (TileLang cache, `torch.utils.cpp_extension.load` without the hook, Triton);
- you want inputs the drivers cannot produce, or a build that takes seconds instead of a model load.

Skip the harness if the driver already builds with `-lineinfo` *and* iterating on it is fast enough.

---

## What a good harness contains

1. The kernel (verbatim copy of the device code + any `__device__` helpers it calls).
2. Explicit template instantiations for every template-parameter combination you plan to profile (e.g. `<TILE_M, TILE_N>`, `<VEC_WIDTH, BLOCK_SIZE>`, whatever the kernel is parameterized on).
3. Optional input loading — from a binary file, safetensors, or synthetic.
4. A minimal `main()` that parses CLI args, allocates GPU memory, launches the kernel, synchronizes, and exits.

Things that should NOT be in the harness:

- Framework dependencies (torch, TVM, pybind11) — they slow the build and create noise in the profile.
- Multi-kernel pipelines — profile each kernel separately unless measuring kernel-to-kernel interactions.
- Repeated warmup / timing loops — ncu replays automatically, so run the kernel exactly once (with `-c 1`).
- Correctness checks — verify correctness separately, don't couple it to profiling.

---

## Template

A complete reusable template lives at [`../scripts/harness_template.cu`](../scripts/harness_template.cu). Customize these sections:

1. **Replace `KERNEL_INCLUDE_GOES_HERE`** with `#include` or paste the kernel source.
2. **Add explicit instantiations** for every template parameter combination you want to profile.
3. **Define the input shape parameters** (grid/block sizes, tensor shapes, any knobs).
4. **Fill in `alloc_and_fill()`** to allocate/initialize inputs correctly for your kernel.
5. **Fill in `launch_kernel()`** to do the actual kernel launch with the right arguments.

Compile with:
```bash
nvcc -ccbin "$(command -v g++)" -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 -lineinfo \
     -I third_party/cutlass/include harness.cu -o harness -lcuda
```

`sm_90a` (not `sm_90`) is what unlocks wgmma and TMA; `-lcuda` is needed for `cuTensorMapEncodeTiled`. Compile with a compatible CUDA toolchain; execution requires the intended GPU. For another GPU, replace the gencode (check `nvidia-smi --query-gpu=compute_cap --format=csv`).

---

## Real data vs synthetic data

There are two useful levels of fidelity for harness inputs:

### Shape-matched: Random-but-reasonable synthetic (shape-matched)

`std::uniform_real_distribution` with sensible ranges (e.g., weights in `[-0.5, 0.5]`, probabilities in `[0, 1]`). Set the *exact* shape (all variable axes of the workload) to match a specific real instance from the dataset.

**Use when:** the kernel has no data-dependent branches that materially affect perf, but you want stable inputs. This is the default for most perf profiling.

Example pattern:
```cpp
fill_bf16_random(h_input_main, 0xA0A0ULL, 0.5f);   // main activations in [-0.5, 0.5]
fill_bf16_random(h_input_small, 0xD0D0ULL, 0.25f); // smaller-magnitude side input
fill_f32_random(h_params, 0x22222ULL, 1.0f);       // parameters — any range that avoids NaN/Inf
for (auto& x : h_params) x = -1.0f - std::fabs(x); // squash into a specific sign/range if the kernel requires it
```

### Actual workload: Actual dataset tensors (real safetensors)

Load the exact BF16/F32 bytes from a `.safetensors` file shipped with the workload.

**Use when:**
- The kernel has branches that might depend on input values (e.g., an early-exit on a magnitude threshold, or a special-case path for denormals / large values).
- The user explicitly asks to profile with real data ("必须 load real workload").
- You're comparing against a reference implementation's output for correctness.

A header-only safetensors reader (no external deps) lives at [`../scripts/safetensors_loader.h`](../scripts/safetensors_loader.h). It parses the 8-byte header length + JSON header + raw tensor bytes — everything a safetensors file ships.

Example:
```cpp
#include "safetensors_loader.h"

SafetensorsFile st = SafetensorsFile::load("/path/to/workload.safetensors");
const uint8_t* input_bytes = st.tensor_bytes("<input_tensor_name>");
std::memcpy(h_input.data(), input_bytes, n_elems * sizeof(<dtype>));

// Shapes are parsed from the header — read what the definition says is variable:
int axis_0 = (int)st.entry("<input_tensor_name>").shape[0];
int axis_1 = (int)st.entry("<other_tensor_name>").shape[0] - 1;  // etc.
```

This is free relative to compilation time and removes all doubt about data-dependent effects.

---

## Choosing representative workloads

If the user's dataset has many workloads, you cannot profile them all. Pick 2-3 workloads that together cover:

1. **Each active dispatch path.** If the kernel's host-side dispatcher picks different template instantiations / grid configs based on input shape, profile one workload per path. Identify the dispatch rules by reading the launcher code, not by guessing.
2. **The largest realistic workload** in the hot-path dispatch — usually the most performance-sensitive.
3. **A worst-case-imbalance workload** if the kernel has a variable-length inner loop (one where different CTAs perform different amounts of work based on the input). Pick an input whose per-CTA work distribution has a high max/min ratio — that's your tail-effect probe.

Example selection approach (for a kernel whose dispatcher picks between two template instantiations by batch size):
- A canonical large-batch workload — exercises the primary dispatch path.
- A small-batch workload — small grid, often reveals SM idleness or under-fill.
- If the large-batch workload has highly uneven per-element work (check the relevant axis in the dataset), that's your tail-effect probe; if every element has the same work, hunt for a separately-imbalanced workload.

### Where the shapes come from here

The Target declares its shapes (the model's `spec.py` and the Target's `target.py`), and the parity drivers already run them — there is no dataset to browse. For a standalone harness, copy the shape tuple from the Target and, when data-dependent branches matter (early exits, masks), dump the real tensors from a parity run and load them with `safetensors_loader.h`.

If a future workload *does* come as a dataset (e.g. a flashinfer-trace tree with `definitions/`, `workloads/*.jsonl`, `blob/*.safetensors`), the upstream skill ships a browser for it (`helpers/list_flashinfer_workloads.py` in mit-han-lab/ncu-report-skill); the principle is the same — learn the schema, pick one instance per dispatch path plus the largest and the most imbalanced, and reference tensors by absolute path.

---

## Explicit template instantiation

If the kernel is a template, you must force the compiler to emit each variant you'll profile. Without this, instantiations that aren't used by `main()` will be stripped, and ncu's `-k "regex:..."` won't find them.

```cpp
template __global__ void my_kernel<8, 256>(
    const __nv_bfloat16*, const __nv_bfloat16*, /* ... other args ... */,
    float*, float*);

template __global__ void my_kernel<4, 256>(
    const __nv_bfloat16*, const __nv_bfloat16*, /* ... other args ... */,
    float*, float*);
```

The launch site in `main()` picks the right instantiation based on a CLI flag.

---

## Sanity check before profiling

Always run the harness once without ncu to confirm it launches correctly:

```bash
./harness --workload /path/to/workload.safetensors
# expected stderr (exact text depends on the harness you wrote):
# [harness] loaded workload: <axis1>=... <axis2>=...
# [harness] grid=(...) block=(...) launching <variant>...
# [harness] done.
```

If it crashes or hangs, fix that *before* adding ncu to the mix — ncu errors are far less descriptive than plain CUDA runtime errors.

If feasible, also spot-check correctness with `cuda-memcheck` or a golden-output test — but not inside the profiling harness itself.

---

## When NOT to harness

Sometimes the kernel's perf genuinely depends on surrounding code — e.g., the kernel reuses a specific L2 state set up by a prior kernel, or the kernel's launch configuration depends on a runtime dispatch step. In that case profile through the original binary (even if it's slower to iterate) and make sure the build system has `-lineinfo`. Check the nvcc invocation with `ninja -v` or `make V=1`.

Alternatively, you can build a harness that runs the *prior* kernel too, to reproduce the right L2 state. But this is rare — most kernels are essentially independent of prior state once a warmup pass has happened.
