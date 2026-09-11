# Candidate plans

Call-site plans tried during optimization of the H100 Targets, kept for A/B
runs. None of them is a deployment configuration: the shipped plan and the
reference plan live on the Target (`hardware/nvidia/h100/<model>/target.py`).
Pass one to any harness as `--plan lab/plans/<name>.json`.

A file is named `<target>-<name>.json`, where `<target>` is the Target's short
alias (`pi05`, `pi0`; `src/flash_vla/inference.py`): `python -m tests.targets` reads
the Target from that prefix to check that the plan binds, so a plan whose
prefix names no Target fails the smoke check.

| file | route |
|---|---|
| `pi0-siglip-cublas.json` | Pi0's two pre-norm vision projections on the shared cuBLASLt form |
| `pi05-siglip-cublas.json` | the same route on Pi0.5, where its TileLang backend already reaches it (the A/B control) |
| `pi0-siglip-shipped-ln.json` / `pi0-siglip-shipped-cuda.json` | Pi0's shipped route plus the vision change, so a gate against `shipped` measures what deployment would see |
| `pi05-siglip-shipped-attn.json` | Pi0.5's shipped route plus the fused vision attention, same purpose |
| `pi0-siglip-attn.json` / `pi05-siglip-attn.json` | the fused vision attention kernel alone |
| `pi0-siglip-ln.json` / `pi05-siglip-ln.json` | the hand-written vision LayerNorm ahead of the two cuBLASLt projections |
| `pi0-siglip-cuda.json` / `pi05-siglip-cuda.json` | both, the combined vision route |
| `pi05-attn-cuda.json` | CUDA encoder attention + CUDA decoder attention pair |
| `pi05-ffn-cuda.json` | CUDA encoder attention + CUDA FFN pair |
| `pi05-ffn-cuda-fused-producer.json` | plus the fused out-projection producer |
| `pi05-attn-ffn-cuda.json` | attention pair and FFN pair on CUDA |
| `pi05-attn-ffn-cuda-fused-producer.json` | the shipped route without the PDL chain |
| `pi05-attn-ffn-cuda-fused-producer-enc-tilelang.json` | shipped decoder route, encoder attention on the torch chain (the encoder A/B leg) |
| `pi05-attn-ffn-cuda-fused-producer-pdlffn.json` | PDL chain on the FFN half only |
| `pi05-gemma-cuda.json` | the shipped Pi0.5 route with the backbone attention taken from the shared `gemma_backbone` component instead of the Target's own copy |
| `pi0-preshipped.json` | Pi0's shipped route as it stood before the Gemma backbone package: the three action-expert fusions and nothing else. It exists so the promotion gate's latency legs can compare the candidate against what shipped WAS, once the shipped plan itself has moved |
| `pi05-preshipped.json` | the same for Pi0.5: backbone attention on the Target's own `cuda` copy, the five action-expert sites on `cuda-pdl` |
| `lingbot_vla-packed-projections.json` | the expert's q/k/v and gate/up as one packed GEMM each, and attention without the repeated key/value copies |
| `lingbot_vla-expert-loop.json` | plus the Target's own denoising loop over a resident key/value cache |
| `lingbot_vla-fused-rope.json` | plus the hand-written projection epilogue (widen, rotate, write the cache) |
| `lingbot_vla-fused-attention.json` | plus the head-major cache, the fused masked softmax and the attention epilogue |
| `lingbot_vla-fused-prefix.json` | plus the Target's own backbone prefix pass on those kernels |
| `lingbot_vla-aligned-vision.json` | plus the width-aligned packed vision feed-forward |
| `lingbot_vla-vision-norm.json` | plus the single-launch vision RMSNorm |
| `lingbot_vla-fused-mlp.json` | plus the AdaRMS that absorbs its residual add and the fused gated activation |
| `lingbot_vla-fused-backbone.json` | plus the same pointwise kernels and a packed gate/up in the backbone. It exists so a later candidate can A/B against what shipped WAS |
| `lingbot_vla-skinny-projections.json` | plus the expert's two 768-wide output projections on the hand-written weight-stationary GEMM, whose split-K partials reduce through distributed shared memory. This is what the LingBot Target ships. The packed q/k/v and gate/up projections are deliberately NOT here: the same kernel measured 5.12 and 5.91 us against cuBLAS's 4.79 and 5.84 on those shapes |
| `lingbot_vla-attention-kernel.json` | the flash-form attention kernel of `cuda/kernels/attention.cu`. **Measured 7x slower than the route it replaces** (results/lingbot-h100/run-01); kept so the kernel stays reachable for the split-key rewrite it needs |
| `pi0-gemma-cuda.json` | the shipped Pi0 route plus the backbone attention and the output projection on the shared `gemma_backbone` component. The FFN down projection is deliberately NOT here: cuBLAS measured 6.6 us per call slower than Pi0's TileLang body in the graph, where the hidden buffer it reads is L2-resident (job 599832) |
