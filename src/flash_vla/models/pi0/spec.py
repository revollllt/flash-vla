"""Hardware-independent Pi0 constants and checkpoint schema."""

RANDOM_CHECKPOINT_REVISION = "flash-vla/pi0-random-checkpoint/v1"


def random_checkpoint_revision(seed: int) -> str:
    """Immutable project ID for the deterministic synthetic checkpoint fixture."""
    return f"{RANDOM_CHECKPOINT_REVISION}/seed-{seed}"

IMAGE_SIZE = 224
IMAGE_CHANNELS = 3
VISION_LAYERS = 27
VISION_TOKENS = 256
VISION_DIM = 1152
VISION_FFN = 4304
VISION_HEADS = 16
VISION_HEAD_DIM = 72
ENCODER_LAYERS = 18
ENCODER_DIM = 2048
ENCODER_FFN = 16384
HEAD_DIM = 256
DECODER_HEADS = 8
KV_HEADS = 1
QKV_WIDTH = 2560
DECODER_DIM = 1024
DECODER_FFN = 4096
STATE_DIM = 32
ACTION_DIM = 32
ROPE_THETA = 10000


def weight_shapes(prompt_len: int) -> dict[str, tuple[int, ...]]:
    """Expected checkpoint tensors for a Pi0 model with this prompt length."""
    return {
        "vision_patch_embedding_w": (14, 14, 3, 1152), "vision_patch_embedding_b": (1152,),
        "vision_position_embedding": (256, 1152),
        "vision_attn_qkv_w": (VISION_LAYERS, 1152, 3 * 1152),
        "vision_attn_qkv_b": (VISION_LAYERS, 3 * 1152),
        "vision_attn_o_w": (VISION_LAYERS, 1152, 1152),
        "vision_attn_o_b": (VISION_LAYERS, 1152),
        "vision_ffn_up_w": (VISION_LAYERS, 1152, 4304),
        "vision_ffn_up_b": (VISION_LAYERS, 4304),
        "vision_ffn_down_w": (VISION_LAYERS, 4304, 1152),
        "vision_ffn_down_b": (VISION_LAYERS, 1152),
        "vision_pre_attn_norm_w": (VISION_LAYERS, 1152),
        "vision_pre_attn_norm_b": (VISION_LAYERS, 1152),
        "vision_pre_ffn_norm_w": (VISION_LAYERS, 1152),
        "vision_pre_ffn_norm_b": (VISION_LAYERS, 1152),
        "vision_final_norm_w": (1152,), "vision_final_norm_b": (1152,),
        "encoder_multi_modal_projector_w": (1152, 2048),
        "encoder_multi_modal_projector_b": (2048,),
        "encoder_attn_qkv_w": (ENCODER_LAYERS, 2048, 2560),
        "encoder_attn_o_w": (ENCODER_LAYERS, 2048, 2048),
        "encoder_ffn_gate_w": (ENCODER_LAYERS, 2048, 16384),
        "encoder_ffn_up_w": (ENCODER_LAYERS, 2048, 16384),
        "encoder_ffn_down_w": (ENCODER_LAYERS, 16384, 2048),
        "decoder_state_in_proj_w": (32, 1024), "decoder_state_in_proj_b": (1024,),
        "decoder_action_fused_in_proj_w": (32, 1024),
        "decoder_action_fused_time_biases": (10, 1024),
        "decoder_action_mlp_w": (1024, 1024), "decoder_action_mlp_b": (1024,),
        "decoder_attn_qkv_w": (ENCODER_LAYERS, 1024, 2560),
        "decoder_attn_o_w": (ENCODER_LAYERS, 2048, 1024),
        "decoder_ffn_gate_w": (ENCODER_LAYERS, 1024, 4096),
        "decoder_ffn_up_w": (ENCODER_LAYERS, 1024, 4096),
        "decoder_ffn_down_w": (ENCODER_LAYERS, 4096, 1024),
        "decoder_action_fused_out_proj_w": (1024, 32),
        "decoder_action_fused_out_proj_b": (32,),
        "language_embeds": (prompt_len, 2048),
    }


def source_weight_shapes() -> dict[str, tuple[int, ...]]:
    """OpenPI PyTorch parameter ABI, independent of runtime folding and schedules."""
    shapes = {
        'action_in_proj.bias': (1024,),
        'action_in_proj.weight': (1024, 32),
        'action_out_proj.bias': (32,),
        'action_out_proj.weight': (32, 1024),
        'action_time_mlp_in.bias': (1024,),
        'action_time_mlp_in.weight': (1024, 2048),
        'action_time_mlp_out.bias': (1024,),
        'action_time_mlp_out.weight': (1024, 1024),
        'paligemma_with_expert.gemma_expert.lm_head.weight': (257152, 1024),
        'paligemma_with_expert.gemma_expert.model.norm.weight': (1024,),
        'paligemma_with_expert.paligemma.lm_head.weight': (257152, 2048),
        'paligemma_with_expert.paligemma.model.language_model.norm.weight': (2048,),
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias': (2048,),
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight': (2048, 1152),
        'paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias': (1152,),
        'paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight': (1152, 3, 14, 14),
        'paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight': (256, 1152),
        'paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias': (1152,),
        'paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight': (1152,),
        'state_proj.bias': (1024,),
        'state_proj.weight': (1024, 32),
    }
    for layer in range(ENCODER_LAYERS):
        prefix = 'paligemma_with_expert.gemma_expert.model.layers.' + str(layer) + "."
        shapes.update({
            prefix + 'input_layernorm.weight': (1024,),
            prefix + 'mlp.down_proj.weight': (1024, 4096),
            prefix + 'mlp.gate_proj.weight': (4096, 1024),
            prefix + 'mlp.up_proj.weight': (4096, 1024),
            prefix + 'post_attention_layernorm.weight': (1024,),
            prefix + 'self_attn.k_proj.weight': (256, 1024),
            prefix + 'self_attn.o_proj.weight': (1024, 2048),
            prefix + 'self_attn.q_proj.weight': (2048, 1024),
            prefix + 'self_attn.v_proj.weight': (256, 1024),
        })
    for layer in range(ENCODER_LAYERS):
        prefix = 'paligemma_with_expert.paligemma.model.language_model.layers.' + str(layer) + "."
        shapes.update({
            prefix + 'input_layernorm.weight': (2048,),
            prefix + 'mlp.down_proj.weight': (2048, 16384),
            prefix + 'mlp.gate_proj.weight': (16384, 2048),
            prefix + 'mlp.up_proj.weight': (16384, 2048),
            prefix + 'post_attention_layernorm.weight': (2048,),
            prefix + 'self_attn.k_proj.weight': (256, 2048),
            prefix + 'self_attn.o_proj.weight': (2048, 2048),
            prefix + 'self_attn.q_proj.weight': (2048, 2048),
            prefix + 'self_attn.v_proj.weight': (256, 2048),
        })
    for layer in range(VISION_LAYERS):
        prefix = 'paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.' + str(layer) + "."
        shapes.update({
            prefix + 'layer_norm1.bias': (1152,),
            prefix + 'layer_norm1.weight': (1152,),
            prefix + 'layer_norm2.bias': (1152,),
            prefix + 'layer_norm2.weight': (1152,),
            prefix + 'mlp.fc1.bias': (4304,),
            prefix + 'mlp.fc1.weight': (4304, 1152),
            prefix + 'mlp.fc2.bias': (1152,),
            prefix + 'mlp.fc2.weight': (1152, 4304),
            prefix + 'self_attn.k_proj.bias': (1152,),
            prefix + 'self_attn.k_proj.weight': (1152, 1152),
            prefix + 'self_attn.out_proj.bias': (1152,),
            prefix + 'self_attn.out_proj.weight': (1152, 1152),
            prefix + 'self_attn.q_proj.bias': (1152,),
            prefix + 'self_attn.q_proj.weight': (1152, 1152),
            prefix + 'self_attn.v_proj.bias': (1152,),
            prefix + 'self_attn.v_proj.weight': (1152, 1152),
        })
    return shapes

# Architecture ABI, before runtime scheduling and checkpoint-value transforms.
from flash_vla.runtime.identity import inference_signature

MODEL_REVISION = 'pi0-r1'
INFERENCE_CONTRACT = {
    "architecture": {
        "family": 'pi0',
        "vision": [VISION_LAYERS, VISION_DIM, VISION_FFN, VISION_HEADS, VISION_HEAD_DIM],
        "backbone": [ENCODER_LAYERS, ENCODER_DIM, ENCODER_FFN],
        "expert": [ENCODER_LAYERS, DECODER_DIM, DECODER_FFN],
        "attention": [DECODER_HEADS, KV_HEADS, HEAD_DIM, ROPE_THETA],
        "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
    },
    "parameter_shapes": source_weight_shapes(),
    "weight_layout": 'openpi-pytorch-out-in-v1',
    "io_contract": {
        "images": ["num_views", IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS],
        "state": [STATE_DIM], "noise": ["chunk", ACTION_DIM],
        "actions": ["chunk", ACTION_DIM],
    },
    "control_flow": {'state': 'continuous-suffix-token', 'expert_norm': 'rms', 'denoise': 'euler', 'prefix': 'bidirectional', 'timestep': 'action-time-mlp'},
}
INFERENCE_SIGNATURE = inference_signature(**INFERENCE_CONTRACT)
