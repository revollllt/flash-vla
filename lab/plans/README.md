# Candidate plans

Call-site plans tried during optimization of the H100 Targets, kept for A/B
runs. None of them is a deployment configuration: the shipped plan and the
reference plan live on the Target (`hardware/nvidia/h100/<model>/target.py`).
Pass one to any harness as `--plan lab/plans/<name>.json`.

A file is named `<target>-<name>.json`, where `<target>` is the Target's short
alias (`pi05`, `pi0`; `benchmarks/targets.py`): `python -m eval.smoke` reads
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
