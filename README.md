# flash-vla

Extreme-latency-optimized vision-language-action (VLA) inference, built as a
hardware/model co-design.

The whole Pi0 forward pass — 27 vision layers, 18 prefix-encoder layers, and 18
decoder layers per diffusion step — runs as kernels captured into a single CUDA
graph. `forward()` copies three inputs into static buffers and replays.

## Design philosophy

This is not a general-purpose inference framework. The atomic production unit is
a *target*:

> Target = device profile × model revision × shape profile × precision policy

A target owns its pipeline, static buffer plan, per-call-site kernel bindings,
fusion boundaries, and tuning results. Everything is specialized to a fixed
workload and measured inside the CUDA graph, at the real shape, with cold
weights — the only regime these latency numbers mean anything in.

**TileLang is the current main line, not the destination.** It is a
performance/efficiency trade-off for agile development. To reach the remaining
performance on a given device, the project will also use hand-written CUDA and
other means. The `backends/` layout is designed for exactly that: each backend
exposes the same wrapper registry, and a single pipeline can mix kernels from
different backends per call site (`op_table(plan={...})`).

The co-design ethos extends to the hardware roadmap: the H100 × Pi0 target is
the first deployment, with RTX 5090 / RTX 4090 targets planned.

## Install

```
pip install -e .
```

Torch is intentionally unpinned so the CUDA build can match your driver; TileLang
is pinned to 0.1.11, which the kernels depend on for specific lowering behaviour
(see *Constraints that bite*). Developed against torch 2.11.0+cu130 on an
H100 SXM5 with driver 610.43.02.

## Use

```python
from flash_vla import Pi0Inference, random_checkpoint

engine = Pi0Inference(random_checkpoint(), num_views=3, chunk_size=50)
actions = engine.forward(images, state, noise)
```

`random_checkpoint()` fabricates weights so the pipeline can be run and timed
without a trained model. For real weights, pass a dict matching
`flash_vla.models.pi0.spec.weight_shapes()`.

## Benchmarks

```
python -m benchmarks e2e --prompt-len 0
python -m benchmarks profile --compare
python -m eval.correctness.pi0.fused_vs_unfused --steps 1
```

All three need a GPU. On a Slurm cluster, submit them rather than running on a
login node — TileLang compiles against the local device and re-reads the source
at compile time.

## Layout

| path | responsibility |
|---|---|
| `src/flash_vla/models/pi0/` | hardware-independent checkpoint schema |
| `src/flash_vla/runtime/cuda/` | graph-safe runtime mechanisms |
| `src/flash_vla/hardware/nvidia/h100/pi0/` | complete H100/Pi0 execution target |
| `.../pi0/backends/` | backend registry: one module per implementation strategy |
| `.../pi0/backends/tilelang/` | TileLang backend: wrappers, fusions, and kernels |
| `benchmarks/` | timing, profiling, and autotune harnesses |
| `eval/` | numerical correctness and policy-quality evaluation |

The target is the atomic performance unit: hardware, model, shape profile, and
precision jointly own the pipeline, buffers, fusion boundaries, and kernels.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the dependency and promotion rules.

**Backend mixing.** Each backend exposes `ALL_WRAPPERS` / `FUSED_WRAPPERS` —
flat dicts of call-site wrappers with identical signatures. `ops.op_table()`
builds the table the pipeline runs against, selecting a backend per call site:

```python
# tilelang for everything, fused decoder on
ops = op_table(fused=True, backend="tilelang")

# tilelang everywhere except decoder attention, which a future CUDA backend owns
ops = op_table(fused=True, plan={"decoder_attention": "cuda"})
```

Selecting the fused decoder is a different operation table, not different
control flow: `op_table(fused=True)` overlays the fused entries onto the
standard map, and the target pipeline calls whatever it is handed.

## Measured

H100 SXM5, 3 views, empty prompt, chunk 50, 10 diffusion steps, 18 layers
(encoder M=768, decoder keys=819). Full-graph wall clock, median of 30:

| configuration | ms |
|---|---:|
| unfused | 16.46 |
| fused | **15.18** |

Kernel self-time totals 15.07 ms of that, so graph overhead — input copies plus
the launch — is under 1%.

Stage split of kernel time: decoder 47%, encoder 37%, vision 16%. Against the
analytic H100 roofline for the same shapes (5.77 ms) the pipeline sits 2.6x
above the floor, and what remains is structural rather than tuning: the encoder
gated FFN is compute-bound at 60% MFU with its stage count capped by keeping
both weights resident, the decoder gate is memory-bound streaming a 16.8 MB
per-layer weight, and every other decoder kernel is under one wave and therefore
latency-bound.

For reference, this pipeline was developed against the upstream realtime-vla
Triton implementation and measured at **1.37x** its full-graph wall clock at the
same shapes (20.83 ms → 15.19 ms) before that dependency was dropped. The
decoder alone was 1.48x.

## Where the speed comes from

Roughly in order of contribution:

**Tile configs measured in the right regime.** Every config came from a sweep at
the call site's real shape, timed inside a CUDA graph with cold weights. This
matters more than it sounds: three configs were wrong in ways an eager benchmark
physically cannot see, because the ~15 us launch overhead hides a 3 us kernel
difference. The worst was a norm kernel launching one block on a 132-SM GPU.

**Warp specialization as a per-call-site choice.** TileLang lowers `T.copy` to
TMA plus a producer/consumer warp split, which pays off above one wave and costs
warps and mbarrier traffic below it. The decoder runs at M=51 — under one wave
everywhere — so its GEMMs want it off, while the encoder and vision stages at
M=768 want it on. Three kernels therefore exist as two variants over one shared
body, differing only in a compile flag.

**Writing through output parameters.** Kernels take their destination as a
parameter instead of allocating and returning, which removes one
device-to-device copy node per call. Converting the last of these cut copy time
in the graph from 0.40 ms to 0.04 ms.

**Lazy pre-norm fusion.** Row scaling commutes with the GEMM reduction, so the
RMS factor can be computed inside the consuming GEMM from the tile it is already
reading. Removes a launch per call site and a serializing bf16 scale from the
mainloop: the decoder gate dropped 43%, the out-projection 69%.

**FlashDecoding attention.** The decoder's scores/softmax/attn@V chain becomes a
split plus a combine with the score matrix never leaving SRAM. Multi-query
attention is what makes the flat form work — all query heads share one KV head,
so tiling the flattened token×head axis is already the head split. 1.43x on the
attention pair.

## Constraints that bite

Things that fail quietly, or fail far from their cause:

- **`tl_scaled_gate`'s tile config is locked.** Correctness there is
  tiling-dependent; some tilings, `BLOCK_M=32` among them, produce garbage
  rather than an error. Re-validate numerically after any re-tune.
- **The gated-FFN kernels require warp specialization.** The dual GEMM reuses
  one shared tile across the two weight stages, which the no-WS pipeline planner
  rejects at compile time. An autotuner sweeping the flag must tolerate that.
- **Every expression inside a `T.Pipelined` body must be inlined.** A named
  temporary lowers to a bind statement the warp-specialization role pass cannot
  classify, and the compile aborts with an internal check failure.
- **A fully out-of-bounds TMA box reads garbage, not zeros.** Only the tail of a
  box that *starts* in bounds is zero-filled. FlashDecoding shrinks its split
  count until no split starts past the key count for exactly this reason; the
  failure mode was NaNs that the running-max update then hid.
- **The FlashDecoding combine tile must stay ≤ 4**, and the split tile must be
  64 — smaller split tiles hit a fragment-layout conflict that fails to lower.
- **Nothing may allocate inside the graph.** Scratch comes from the pool, which
  is frozen after warmup so a miss raises instead of allocating during capture.
- **Do not edit these files while a benchmark job is running.** TileLang
  re-reads the source at compile time, so saving a shorter file mid-run makes
  the compile fail with an out-of-range line number.

## Requirements

H100 (Hopper WGMMA and TMA), TileLang 0.1.11, PyTorch with CUDA. The vision and
encoder attention stay in torch: both are full bidirectional attention over a
long sequence, where cuDNN is already at the roofline. Only the decoder's
multi-query attention over a KV cache has a TileLang implementation.

`num_views == 2` is not supported (upstream's two-part vision branch).

## Provenance

Extracted from a larger H100 megakernel research repository, where this pipeline
was developed as a port of the Triton kernels in the realtime-vla Pi0 inference
implementation. Every kernel was validated op-by-op against that implementation
before the dependency was dropped; what remains in-tree is the fused-vs-unfused
regression gate
(`python -m eval.correctness.pi0.fused_vs_unfused --steps 1`).

Source commit of the parent repository: `a53bcf9`.

## License

MIT — see [LICENSE](LICENSE).
