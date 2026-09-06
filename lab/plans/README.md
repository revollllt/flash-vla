# Candidate plans

Call-site plans tried during optimization of the H100 / Pi0.5 Target, kept
for A/B runs. None of them is a deployment configuration: the shipped plan and
the reference plan live on the Target (`hardware/nvidia/h100/pi05/target.py`).
Pass one to any harness as `--plan lab/plans/<name>.json`.

| file | route |
|---|---|
| `pi05-attn-cuda.json` | CUDA encoder attention + CUDA decoder attention pair |
| `pi05-ffn-cuda.json` | CUDA encoder attention + CUDA FFN pair |
| `pi05-ffn-cuda-fused-producer.json` | plus the fused out-projection producer |
| `pi05-attn-ffn-cuda.json` | attention pair and FFN pair on CUDA |
| `pi05-attn-ffn-cuda-fused-producer.json` | the shipped route without the PDL chain |
| `pi05-attn-ffn-cuda-fused-producer-enc-tilelang.json` | shipped decoder route, encoder attention on the torch chain (the encoder A/B leg) |
| `pi05-attn-ffn-cuda-fused-producer-pdlffn.json` | PDL chain on the FFN half only |
