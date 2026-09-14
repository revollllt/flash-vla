# Pi0.5 RTX 5090 backbone QKV fusion

Hypothesis: replace repeated RMSNorm and RoPE/scatter traversals with two CUDA
stages around the unchanged bf16 torch.mm. The original call-site profile
attributed 2.329 ms and 414 launches over 18 layers, suggesting 0.7–1.2 ms
recoverable time before integration.

The candidate reuses this Target's `fused_backbone` RMSNorm ABI. It writes a
bf16 normalized activation before the 2048x2560 GEMM, then reads the bf16
projection in a hand-written RoPE/scatter kernel. This ordering is the
backbone's torch semantics. A flat grid processes adjacent bf16 pairs and
repeats the 256-wide cosine/sine table across eight Q heads and one K head;
V is copied directly. The grid caps at 680 CTAs, matching the existing Target
streaming pointwise choice, without a new tile sweep. RoPE multiplication and
addition compile with `--fmad=false`.

Native libraries load on the first invocation during warmup. Constructing the
wrapper with CUDA_HOME/FLASH_VLA_NVCC unset succeeds without invoking a compiler.
The per-factory scratch projection is 968x2560 bf16, 4,956,160 bytes. No existing
library, registry or Target declaration is changed.

Validation on the exclusive RTX 5090, seed 42, checkpoint
`kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha`:

- CUDA compilation and capture passed.
- Actual backbone inputs were cloned before each call. Layers 0, 9 and 17
  produced bit-exact Q, K, V and x_norm outputs: max_abs=0 and rel_rms=0.
- Thirty graph samples per route, each cycling all 18 distinct layer weights:
  torch median 137.9600 us/call; fused median 58.0524 us/call. This is a local
  reduction of 57.92%, or 1.4383 ms across 18 layers. Deployed latency remains
  the model integration loop's independent measurement.
- Raw samples and workload identity are retained at
  `artifacts/rtx5090-pi05/gpt6-prefix-qkv-local.json` in the main checkout.

Reproduce from the candidate checkout in the target's CUDA/Python environment:

```bash
PYTHONPATH="$PWD/src:$PWD" python -m lab.pi05.rtx5090_prefix_qkv --seed 42 \
  --option converted_checkpoint=<converted-checkpoint-directory> \
  --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --output artifacts/rtx5090-pi05/gpt6-prefix-qkv-local.json
```
