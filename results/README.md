<!-- lab.results: generated below -->

# Flash-VLA results

Performance comparisons are local to the representative checkpoint/fixture segment.

## Current optimization runs

- [lingbot-h100/run-01](lingbot-h100/run-01/README.md) — Deployed end-to-end latency of `h100/lingbot_vla` on its real post-training checkpoint and frozen seed-42 fixture, starting from the current `shipped` plan. The objective is the deployed `chunk_latency` median.
- [lingbot-h100/run-02](lingbot-h100/run-02/README.md) — Continues run-01 from its final version, with an explicit target: reach Pi0.5's 15.864 ms, which a pure roofline comparison says LingBot should be capable of.
- [pi0-rtx5090/run-01](pi0-rtx5090/run-01/README.md) — First optimization run of `rtx5090/pi0`, from the all-torch bring-up route to a hand-written CUDA route: **46.794 → 27.556 ms, 1.70×**.
- [pi05-rtx5090/gpt6-run-01](pi05-rtx5090/gpt6-run-01/README.md) — Active optimization run on branch `gpt6-pi05-5090`, starting at `5ac75bc`.
- [pi05-rtx5090/mxfp8-llm-ffn](pi05-rtx5090/mxfp8-llm-ffn/README.md) — Optimization run of the quantized workload `mxfp8-llm-ffn` on branch `exp/pi05-5090/mxfp8-llm-ffn`, starting at `29a29ea`. The recipe puts the three GEMMs of every LLM-backbone FFN layer (gate, up, down) in MXFP8: E4M3 values with one UE8M0 scale per 32 elements along K, weights quantized once from BF16, activations quantized on the device. Every other call site keeps its BF16 route. The forward runs 17 of the 18 FFN layers: the last one does not reach the prefix KV cache, so the graph omits it in every plan. The recipe was approved on its quality on 408 LIBERO observations (`quant-pi05-ffn-libero`); its math is fixed in `rtx5090/pi05/target.py` (`QUANTIZATION`) and recorded in every identity's `execution_variant.quantization`.

## Retired Campaign entries

Saved traces under `targets/`, from the retired Campaign controller. **These are historical.** The "Current best ms" column is that campaign's last measurement, not a number this repository is working from; the runs above carry those.

| Hardware | Model revision | Shape | Variant | Context | Anchor ms | Current best ms | Speedup | Iter |
|---|---|---|---|---|---:|---:|---:|---:|
| h100-sxm5-80gb | [lingbot-vla-r1](targets/016bff50eb2c9ef0cbd92b17b9da996614e06f1f679306c611f3e822599755a7/README.md) | {"action_dim":75,"backbone_dim":2048,"backbone_ffn_dim":11008,"batch":1,"chunk":50,"expert_dim":768,"expert_ffn_dim":2752,"head_dim":128,"image_height":224,"image_width":224,"kv_heads":2,"language_slots":72,"layers":36,"patch_rows_per_view":256,"patch_width":1176,"prefix_len":264,"query_heads":16,"state_dim":75,"steps":10,"suffix_len":51,"views":3,"vision_dim":1280,"vision_ffn_dim":3420,"vision_head_dim":80,"vision_heads":16,"vision_layers":32,"visual_tokens":192,"visual_tokens_per_view":64} | {"cache":{"mode":"none"},"quantization":{"mode":"bf16"}} | lingbot-vla-4b-posttrain-robotwin@fb71a2c9749ccfedbb7290c2c3f0e5e7c7305c9e+qwen2.5-vl-3b@66285546d2b821cf421d4f5eb2576359d3770cd3 | 86.392 | 64.804 | 1.333× | 6 |
| h100-sxm5-80gb | [pi05-r1](targets/c1091dd42d1dbb818ea69e14605ed706d88b5ead4dae33e75cb7e0ff9bb3e494/README.md) | {"action_dim":32,"batch":1,"chunk":50,"encoder_dim":2048,"encoder_ffn_dim":16384,"expert_dim":1024,"expert_ffn_dim":4096,"expert_tokens":50,"head_dim":256,"image_channels":3,"image_height":224,"image_width":224,"kv_heads":1,"layers":18,"num_views":3,"prefix_len":968,"prompt_len":200,"qkv_width":2560,"query_heads":8,"state_dim":32,"steps":10,"vision_dim":1152,"vision_ffn_dim":4304,"vision_head_dim":72,"vision_heads":16,"vision_layers":27,"visual_tokens":768,"visual_tokens_per_view":256} | {"cache":{"mode":"none"},"quantization":{"mode":"bf16"}} | pravsels/pi05-build-block-tower-baseline@c02537599ca0577dcf8081018a49978e9129be64/checkpoints/50000 | 15.864 | 15.864 | 1.000× | 1 |

<!-- lab.results: end generated; hand-written notes go below -->

Each run is its own comparison context. A different GPU, checkpoint, fixture or
timing condition starts a new run directory, and those curves are never joined
into one speedup: an RTX 5090 number and an H100 number in the list above answer
different questions and do not compare.
