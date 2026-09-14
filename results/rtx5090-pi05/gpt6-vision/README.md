# Vision normalization and activation fusion

Hypothesis: handwritten FP32 LayerNorm into BF16, plus native tanh GELU of
the BF16-rounded addmm output, removes casts and traversals while preserving
the two existing bias-fused torch.addmm calls. Parent deployment-005 profile:
vision norm_ffn_up 2.055 ms / 270 launches / 27 calls; norm_qkv 1.249 ms /
162 launches / 27 calls. Expected model recovery: roughly 0.5-1 ms.

Read this Target's torch definitions, the RTX 5090 Pi0 CUDA LayerNorm/GELU,
and row-traversal-fusion wiki. The new Target-local LayerNorm retains values
through two FP32 reductions (mean, centered variance), avoiding mean-square
subtraction cancellation. Gamma/beta stay FP32 until BF16 normalization store.
FFN addmm writes the BF16 output directly, then handwritten tanh GELU updates
that buffer in place. No GEMM epilogue applies GELU before BF16 rounding.

The partial backend exports only vision_encoder_norm_qkv and
vision_encoder_norm_ffn_up. The runner's scratch owns the normalization
buffer. Imports and make_wrappers do not compile or load native code; the first
real invocation does so during warmup. The registry/Target are unchanged here.

Run on ssh yx5090, RTX 5090, torch 2.13.0+cu130, existing user environment,
with no simultaneous GPU/compilation work:

    source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
    PYTHONPATH=$PWD/src:$PWD /home/ubuntu/flash-vla/.venv/bin/python \
      -m lab.sm120.pi05_vision_fusion_probe \
      --checkpoint /home/ubuntu/models/pi05_belt_cup_pytorch \
      --checkpoint-id kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
      --out results/rtx5090-pi05/gpt6-vision/local.json

The probe records all 27 actual layer input snapshots for each site on the
seed-42 reference trajectory, retaining checkpoint weights. Layer 0/13/26
checks restore reference output before advancing the model. Normalized
activations and final outputs pass existing shallow tolerances. Worst output
rel_rms is 9.0612e-5 with cosine at least 0.9999999958. Layer-26 FFN max_abs
is 0.125 while p99_abs is zero: sparse BF16 rounding changes, not bitwise
identity. Worst norm rel_rms is 1.416e-5.

Timing uses the existing graph helper on its capture stream, cycling all 27
real checkpoint layer weights and snapshotted inputs per graph, 27 warmup
calls, 30 samples and fresh captures in torch/fused/fused/torch order.
The shapes are x(3,256,1152), QKV weight(1152,3456), FFN weight(1152,4304).

| Site | Torch first (ms/27 calls) | Fused first | Fused second | Torch second |
| --- | ---: | ---: | ---: | ---: |
| norm_qkv | 1.319120 | 1.006032 | 1.009888 | 1.327520 |
| norm_ffn_up | 2.183616 | 1.426304 | 1.439136 | 2.185472 |

Raw per-call samples and numerical metrics are in local.json. The two local
chains save approximately 1.07 ms combined under these conditions. Full-model
official accuracy and deployed end-to-end timing remain the parent model
loop's integration work; local savings do not establish deployment gains.
