# GR00T N1.7 LIBERO PyTorch reference

`rtx5090/groot_n17` (alias `groot-n17`) runs the official LIBERO checkpoint
through Flash-VLA's vision, language and action stages. Both `reference` and
`shipped` initially use plain PyTorch, including PyTorch SDPA. There are no
custom kernels, quantization, torch.compile or cross-observation feature caches.
The existing runtime captures each stage in a CUDA Graph.

## Workload

| Setting | Reference workload |
|---|---|
| Checkpoint | `nvidia/GR00T-N1.7-LIBERO`, `libero_10`, revision `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21` |
| Official source | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), revision `51d4c89f72fda44cbf77285c6a8114b52676b8a1` |
| Backbone assets | `nvidia/Cosmos-Reason2-2B`, revision `9ce19a195e423419c349abfc86fd07178b230561` |
| Precision / batch | BF16 / 1 |
| Observations | Official `demo_data/libero_demo`, episode 0; frames 0, 8 and 16 for parity |
| Views | Image and wrist image; each 256×256 after official preprocessing |
| Model inputs | 512 patches of width 1536; 128 visual tokens; sequence length 156 |
| Backbone / action head | 16 Qwen3-VL layers; 32 DiT blocks; 4 denoising steps |
| Internal output | `[1,40,132]` normalized actions |
| Decoded LIBERO output | 16 actions, each with 7 physical action coordinates |

The Target deliberately keeps this checkpoint's depth and denoising schedule.
Use `--steps 0 --layers 0` for full-depth correctness; the generic checker's
default one-layer/one-step diagnostic is not this workload. A different text
sequence length needs a matching prepared fixture and `sequence_length` option.

The benchmark measures **prepared device tensors → complete model action
chunk**, including input staging and all three stages for every fresh
observation. Video decoding, image processing, tokenization, noise generation,
CPU→GPU delivery of prepared inputs and physical-action decoding are outside
that timing. Do not report it as raw-observation-to-robot-action latency.
Stage FLOP estimates count dense matmuls and attention, excluding pointwise ops.

## Environment and assets

The verified environment is Python 3.12, PyTorch 2.9.0+cu128, TorchVision
0.24.0+cu128, Transformers 4.57.3 and Diffusers 0.35.1. Use a separate environment
from existing Pi0.5 work. The model dependency extra is `.[groot-n17]`.

Official preparation/parity additionally needs the NVIDIA source checkout and
its processor dependencies (`albumentations==1.4.18`, `dm-tree`,
`pandas==2.2.3`, `pyarrow`, `av`, `accelerate`). Install that checkout with
`pip install --no-deps -e "$GROOT_UPSTREAM"` after preparing those dependencies;
TensorRT, FlashAttention and training dependencies are unnecessary here.
PyAV/libdav1d decodes the official AV1 demo videos.

Set machine-local paths for `GROOT_CHECKPOINT` (the `libero_10` directory),
`GROOT_BACKBONE` (the Cosmos snapshot), `GROOT_UPSTREAM` and `GROOT_FIXTURES`.
The local backbone path must contain `nvidia/Cosmos-Reason2`, as required by
the upstream model selector. Resolve Hugging Face assets to local directories
before running; Transformers 4.57.3 may still query model metadata for a remote
ID even when offline mode is requested.

Download only the checkpoint's config, processor config, statistics,
embodiment IDs, safetensors index and two safetensors shards. The official
comparison also loads the original Cosmos checkpoint; Flash-VLA inference
loads its backbone weights directly from the complete LIBERO checkpoint.
Hydrate the episode-0 parquet and both videos from the official Git LFS data.

Prepare one fixture at a time from the project root:

```bash
python -m eval.groot_n17.prepare \
  --checkpoint "$GROOT_CHECKPOINT" --backbone "$GROOT_BACKBONE" \
  --dataset "$GROOT_UPSTREAM/demo_data/libero_demo" \
  --output "$GROOT_FIXTURES" --frame 0
```

Repeat with `--frame 8` and `--frame 16` for the independent numerical checks.
The processor is the official implementation in evaluation mode; the saved
fixture includes the real observation and its processed tensors.

Point `FLASH_VLA_ASSETS` at a JSON file mapping these two logical IDs to local
paths (relative paths resolve beside that JSON):

```json
{
  "nvidia/GR00T-N1.7-LIBERO@2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21/libero_10": "/path/to/libero_10",
  "isaac-gr00t@51d4c89/libero-demo/episode-0/frame-0": "/path/to/episode-0-frame-0.pt"
}
```

## Run and verify

```bash
python -m tests.targets --target rtx5090/groot_n17
python -m eval.correctness --target rtx5090/groot_n17 --steps 0 --layers 0
python -m benchmarks latency --target rtx5090/groot_n17 --plan reference \
  --out artifacts/groot-n17/latency.json
python -m tools.profiling.model --target rtx5090/groot_n17 --plan reference \
  --overview --trace-dir artifacts/groot-n17/profile
```

Use separate processes for the official oracle and the project comparison:

```bash
python -m eval.groot_n17.parity official \
  --checkpoint "$GROOT_CHECKPOINT" --backbone "$GROOT_BACKBONE" \
  --fixture "$GROOT_FIXTURES/episode-0-frame-0.pt" \
  --fixture "$GROOT_FIXTURES/episode-0-frame-8.pt" \
  --fixture "$GROOT_FIXTURES/episode-0-frame-16.pt" \
  --output artifacts/groot-n17/official.pt
python -m eval.groot_n17.parity compare \
  --oracle artifacts/groot-n17/official.pt --assets "$FLASH_VLA_ASSETS" \
  --output artifacts/groot-n17/parity.json
```

The comparison checks identical initial noise, internal actions, decoded
actions, finite outputs, repeatability, and eager versus captured execution.
It uses the existing BF16 numerical requirements without changing tolerances.
Three observations passed: maximum relative RMS was 0.001922 internally and
0.002699 after decoding; minimum decoded cosine was 0.9999964. Eager and
captured outputs were identical. This is numerical validation, not a LIBERO
closed-loop success-rate evaluation.

The existing Target checks and full-depth two-runner correctness check passed.
An initial RTX 5090 measurement with 5 warmups and 30 repetitions gave a median
of **24.919 ms**, ranging from **24.895 to 24.930 ms**, under the timing boundary
above. This is one measurement session with unlocked GPU clocks. A separate
profiler overview captured all three stages; its instrumented times are not
used as the latency result.

## Implementation details that preserve the official forward

- GR00T consumes `ForConditionalGeneration.hidden_states[-1]`. In the verified
  Transformers version that is the final decoder output **before** the final
  language RMSNorm. Normalizing it again changes the action output.
- Match the official Conv3d input's `channels_last_3d` layout. With this layout,
  both vision outputs and DeepStack features matched the official tensors.
- Prepare fixed-grid vision position tables during initialization: upstream
  uploads Python-built indices inside forward, which CUDA Graph capture rejects.
- Use fixed-size gather/scatter for DeepStack injection instead of dynamic
  boolean indexing during capture. RoPE positions remain explicit inputs.
- Preserve category-conditioned projections, image/text attention alternation,
  the official action time embedding and all four Euler updates. Initial noise
  is explicit in the project forward.

The adapted inference code in [reference.py](reference.py) is distributed under
the accompanying [Apache-2.0 license](LICENSE). It retains ordinary Transformers
and Diffusers modules so later optimization can replace individual stages.
