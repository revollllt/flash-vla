"""LingBot-VLA-4B dimensions and exact official checkpoint schema."""
from __future__ import annotations

CHECKPOINT_REVISION = (
    "lingbot-vla-4b-posttrain-robotwin@"
    "fb71a2c9749ccfedbb7290c2c3f0e5e7c7305c9e+"
    "qwen2.5-vl-3b@66285546d2b821cf421d4f5eb2576359d3770cd3"
)

VIEWS = 3
IMAGE_SIZE = 224
PATCH_ROWS_PER_VIEW = 256
PATCH_WIDTH = 1176
VISUAL_TOKENS_PER_VIEW = 64
VISION_LAYERS = 32
VISION_DIM = 1280
VISION_FFN = 3420
VISION_HEADS = 16
VISION_HEAD_DIM = 80
LANGUAGE_SLOTS = 72
PREFIX_LEN = 264
STATE_DIM = 75
ACTION_DIM = 75
CHUNK = 50
SUFFIX_LEN = 51
LAYERS = 36
BACKBONE_DIM = 2048
BACKBONE_FFN = 11008
EXPERT_DIM = 768
EXPERT_FFN = 2752
QUERY_HEADS = 16
KV_HEADS = 2
HEAD_DIM = 128


def weight_shapes() -> dict[str, tuple[int, ...]]:
    """All 1,555 tensors in the frozen post-training safetensors checkpoint."""
    root = "model."
    shapes: dict[str, tuple[int, ...]] = {
        root + "state_proj.bias": (EXPERT_DIM,),
        root + "state_proj.weight": (EXPERT_DIM, STATE_DIM),
        root + "action_in_proj.bias": (EXPERT_DIM,),
        root + "action_in_proj.weight": (EXPERT_DIM, ACTION_DIM),
        root + "action_out_proj.bias": (ACTION_DIM,),
        root + "action_out_proj.weight": (ACTION_DIM, EXPERT_DIM),
        root + "action_time_mlp_in.bias": (EXPERT_DIM,),
        root + "action_time_mlp_in.weight": (EXPERT_DIM, 2 * EXPERT_DIM),
        root + "action_time_mlp_out.bias": (EXPERT_DIM,),
        root + "action_time_mlp_out.weight": (EXPERT_DIM, EXPERT_DIM),
    }

    visual = root + "qwenvl_with_expert.qwenvl.visual."
    shapes[visual + "patch_embed.proj.weight"] = (
        VISION_DIM, 3, 2, 14, 14
    )
    for layer in range(VISION_LAYERS):
        base = visual + f"blocks.{layer}."
        shapes.update({
            base + "attn.proj.bias": (VISION_DIM,),
            base + "attn.proj.weight": (VISION_DIM, VISION_DIM),
            base + "attn.qkv.bias": (3 * VISION_DIM,),
            base + "attn.qkv.weight": (3 * VISION_DIM, VISION_DIM),
            base + "mlp.down_proj.bias": (VISION_DIM,),
            base + "mlp.down_proj.weight": (VISION_DIM, VISION_FFN),
            base + "mlp.gate_proj.bias": (VISION_FFN,),
            base + "mlp.gate_proj.weight": (VISION_FFN, VISION_DIM),
            base + "mlp.up_proj.bias": (VISION_FFN,),
            base + "mlp.up_proj.weight": (VISION_FFN, VISION_DIM),
            base + "norm1.weight": (VISION_DIM,),
            base + "norm2.weight": (VISION_DIM,),
        })
    shapes.update({
        visual + "merger.ln_q.weight": (VISION_DIM,),
        visual + "merger.mlp.0.bias": (4 * VISION_DIM,),
        visual + "merger.mlp.0.weight": (4 * VISION_DIM, 4 * VISION_DIM),
        visual + "merger.mlp.2.bias": (BACKBONE_DIM,),
        visual + "merger.mlp.2.weight": (BACKBONE_DIM, 4 * VISION_DIM),
    })

    backbone = root + "qwenvl_with_expert.qwenvl.model."
    shapes[backbone + "embed_tokens.weight"] = (151936, BACKBONE_DIM)
    shapes[backbone + "norm.weight"] = (BACKBONE_DIM,)
    for layer in range(LAYERS):
        base = backbone + f"layers.{layer}."
        shapes.update({
            base + "input_layernorm.weight": (BACKBONE_DIM,),
            base + "post_attention_layernorm.weight": (BACKBONE_DIM,),
            base + "mlp.down_proj.weight": (BACKBONE_DIM, BACKBONE_FFN),
            base + "mlp.gate_proj.weight": (BACKBONE_FFN, BACKBONE_DIM),
            base + "mlp.up_proj.weight": (BACKBONE_FFN, BACKBONE_DIM),
            base + "self_attn.k_proj.bias": (KV_HEADS * HEAD_DIM,),
            base + "self_attn.k_proj.weight": (KV_HEADS * HEAD_DIM, BACKBONE_DIM),
            base + "self_attn.o_proj.weight": (BACKBONE_DIM, QUERY_HEADS * HEAD_DIM),
            base + "self_attn.q_proj.bias": (QUERY_HEADS * HEAD_DIM,),
            base + "self_attn.q_proj.weight": (QUERY_HEADS * HEAD_DIM, BACKBONE_DIM),
            base + "self_attn.v_proj.bias": (KV_HEADS * HEAD_DIM,),
            base + "self_attn.v_proj.weight": (KV_HEADS * HEAD_DIM, BACKBONE_DIM),
        })

    expert = root + "qwenvl_with_expert.qwen_expert.model."
    shapes[expert + "norm.weight"] = (EXPERT_DIM,)
    for layer in range(LAYERS):
        base = expert + f"layers.{layer}."
        shapes.update({
            base + "input_layernorm.weight": (EXPERT_DIM,),
            base + "input_layernorm.beta.bias": (EXPERT_DIM,),
            base + "input_layernorm.beta.weight": (EXPERT_DIM, EXPERT_DIM),
            base + "input_layernorm.gamma.bias": (EXPERT_DIM,),
            base + "input_layernorm.gamma.weight": (EXPERT_DIM, EXPERT_DIM),
            base + "post_attention_layernorm.weight": (EXPERT_DIM,),
            base + "post_attention_layernorm.beta.bias": (EXPERT_DIM,),
            base + "post_attention_layernorm.beta.weight": (EXPERT_DIM, EXPERT_DIM),
            base + "post_attention_layernorm.gamma.bias": (EXPERT_DIM,),
            base + "post_attention_layernorm.gamma.weight": (EXPERT_DIM, EXPERT_DIM),
            base + "mlp.down_proj.weight": (EXPERT_DIM, EXPERT_FFN),
            base + "mlp.gate_proj.weight": (EXPERT_FFN, EXPERT_DIM),
            base + "mlp.up_proj.weight": (EXPERT_FFN, EXPERT_DIM),
            base + "self_attn.k_proj.bias": (KV_HEADS * HEAD_DIM,),
            base + "self_attn.k_proj.weight": (KV_HEADS * HEAD_DIM, EXPERT_DIM),
            base + "self_attn.o_proj.weight": (EXPERT_DIM, QUERY_HEADS * HEAD_DIM),
            base + "self_attn.q_proj.bias": (QUERY_HEADS * HEAD_DIM,),
            base + "self_attn.q_proj.weight": (QUERY_HEADS * HEAD_DIM, EXPERT_DIM),
            base + "self_attn.v_proj.bias": (KV_HEADS * HEAD_DIM,),
            base + "self_attn.v_proj.weight": (KV_HEADS * HEAD_DIM, EXPERT_DIM),
        })
    if len(shapes) != 1555:
        raise AssertionError(f"LingBot schema has {len(shapes)} tensors, expected 1555")
    return shapes


WEIGHT_SHAPES = weight_shapes()
WEIGHT_NAMES = tuple(WEIGHT_SHAPES)
VISION_WEIGHT_NAMES = tuple(name for name in WEIGHT_NAMES if ".qwenvl.visual." in name)
BACKBONE_WEIGHT_NAMES = tuple(name for name in WEIGHT_NAMES if ".qwenvl.model." in name)
EXPERT_WEIGHT_NAMES = tuple(
    name for name in WEIGHT_NAMES
    if name not in set(VISION_WEIGHT_NAMES) | set(BACKBONE_WEIGHT_NAMES)
)

__all__ = [name for name in globals() if name.isupper()] + ["weight_shapes"]


from flash_vla.runtime.identity import inference_signature

MODEL_REVISION = "lingbot-vla-r1"
INFERENCE_CONTRACT = {
    "architecture": {
        "family": "lingbot-vla",
        "vision": [VISION_LAYERS, VISION_DIM, VISION_FFN, VISION_HEADS, VISION_HEAD_DIM],
        "backbone": [LAYERS, BACKBONE_DIM, BACKBONE_FFN],
        "expert": [LAYERS, EXPERT_DIM, EXPERT_FFN],
        "attention": [QUERY_HEADS, KV_HEADS, HEAD_DIM],
        "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
    },
    "parameter_shapes": WEIGHT_SHAPES,
    "weight_layout": "lingbot-official-out-in-v1",
    "io_contract": {
        "pixel_values": ["views", "patch_rows", PATCH_WIDTH],
        "language_tokens": [1, "language_slots"],
        "language_masks": [1, "language_slots"], "image_masks": ["views"],
        "state": [1, STATE_DIM], "noise": [1, "chunk", ACTION_DIM],
        "actions": [1, "chunk", ACTION_DIM],
    },
    "control_flow": {
        "stages": ["vision", "prefix", "action"], "depth_input": False,
        "denoise": "euler", "state": "continuous-suffix-token",
        "expert_norm": "time-conditioned-scale-shift",
    },
}
INFERENCE_SIGNATURE = inference_signature(**INFERENCE_CONTRACT)
