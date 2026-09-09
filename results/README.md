# Flash-VLA results

Performance comparisons are local to the representative checkpoint/fixture segment.

| Hardware | Model revision | Shape | Variant | Context | Anchor ms | Current best ms | Speedup | Iter |
|---|---|---|---|---|---:|---:|---:|---:|
| h100-sxm5-80gb | [lingbot-vla-r1](targets/016bff50eb2c9ef0cbd92b17b9da996614e06f1f679306c611f3e822599755a7/README.md) | {"action_dim":75,"backbone_dim":2048,"backbone_ffn_dim":11008,"batch":1,"chunk":50,"expert_dim":768,"expert_ffn_dim":2752,"head_dim":128,"image_height":224,"image_width":224,"kv_heads":2,"language_slots":72,"layers":36,"patch_rows_per_view":256,"patch_width":1176,"prefix_len":264,"query_heads":16,"state_dim":75,"steps":10,"suffix_len":51,"views":3,"vision_dim":1280,"vision_ffn_dim":3420,"vision_head_dim":80,"vision_heads":16,"vision_layers":32,"visual_tokens":192,"visual_tokens_per_view":64} | {"cache":{"mode":"none"},"quantization":{"mode":"bf16"}} | lingbot-vla-4b-posttrain-robotwin@fb71a2c9749ccfedbb7290c2c3f0e5e7c7305c9e+qwen2.5-vl-3b@66285546d2b821cf421d4f5eb2576359d3770cd3 | 105.409 | 105.409 | 1.000× | 0 |
