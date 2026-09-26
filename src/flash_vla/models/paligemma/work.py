"""Which Pi0 and Pi0.5 call site each op of the shared PaliGemma-with-expert belongs to.

The vision tower, projector, prompt embedding, backbone and action expert of
`reference.PaliGemmaWithExpert` run the same call sites in both models
(`runtime/ops.py`, standard names); each model adds the rules of its own head
(`models/pi0/work.py`, `models/pi05/work.py`). An op no rule matches belongs to
the call site before it: the residual add after a projection, the GELU and
gating inside the MLP, the RoPE after Q/K/V. Attention is told apart from that
RoPE, which runs in the same module, by its matmuls and softmax.
"""
from __future__ import annotations

from flash_vla.runtime.work import CallSiteRule

#: The aten ops of attention itself, as opposed to the RoPE in the same module.
ATTENTION_OPS = r"aten\.(bmm|mm|_softmax|_safe_softmax)"

RULES = (
    CallSiteRule(r"^paligemma\.vision\.embeddings", None, "vision_encoder_patch_embed"),
    CallSiteRule(r"vision\.encoder\.layers\.\d+\.(layer_norm1|self_attn\.[qkv]_proj)$", None,
                 "vision_encoder_norm_qkv"),
    CallSiteRule(r"vision\.encoder\.layers\.\d+\.self_attn$", None, "vision_encoder_attention"),
    CallSiteRule(r"vision\.encoder\.layers\.\d+\.self_attn\.out_proj$", None,
                 "vision_encoder_out_proj_residual"),
    CallSiteRule(r"vision\.encoder\.layers\.\d+\.(layer_norm2|mlp\.fc1)$", None,
                 "vision_encoder_norm_ffn_up"),
    CallSiteRule(r"vision\.encoder\.layers\.\d+\.mlp\.fc2$", None, "vision_encoder_ffn_down_residual"),
    # The engine's vision output is the hidden state before SigLIP's final
    # LayerNorm; the projector call site applies that norm.
    CallSiteRule(r"^paligemma\.(vision\.post_layernorm|projector)$", None, "llm_backbone_projector"),
    CallSiteRule(r"^paligemma\.embedding$", None, "llm_backbone_embed_prompt"),
    CallSiteRule(r"backbone\.layers\.\d+\.(input_layernorm|self_attn\.[qkv]_proj)$", None,
                 "llm_backbone_norm_qkv_rope"),
    CallSiteRule(r"backbone\.layers\.\d+\.self_attn$", ATTENTION_OPS, "llm_backbone_attention"),
    CallSiteRule(r"backbone\.layers\.\d+\.self_attn\.o_proj$", None, "llm_backbone_out_proj_residual"),
    CallSiteRule(r"backbone\.layers\.\d+\.(post_attention_layernorm|mlp\.(gate|up)_proj)$", None,
                 "llm_backbone_norm_gated_ffn"),
    CallSiteRule(r"backbone\.layers\.\d+\.mlp\.down_proj$", None, "llm_backbone_ffn_down_residual"),
    CallSiteRule(r"expert\.layers\.\d+\.(input_layernorm|self_attn\.[qkv]_proj)$", None,
                 "action_expert_norm_qkv_rope"),
    CallSiteRule(r"expert\.layers\.\d+\.self_attn$", ATTENTION_OPS, "action_expert_attention"),
    CallSiteRule(r"expert\.layers\.\d+\.self_attn\.o_proj$", None, "action_expert_out_proj_residual"),
    CallSiteRule(r"expert\.layers\.\d+\.(post_attention_layernorm|mlp\.(gate|up)_proj)$", None,
                 "action_expert_norm_gated_ffn"),
    CallSiteRule(r"expert\.layers\.\d+\.mlp\.down_proj$", None, "action_expert_ffn_down_residual"),
    # The final norm, the output projection and the Euler step after it.
    CallSiteRule(r"^(paligemma\.expert\.norm|head\.action_out_proj)$", None,
                 "action_expert_action_out_proj"),
)

__all__ = ["RULES"]
