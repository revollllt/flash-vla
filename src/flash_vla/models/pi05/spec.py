"""Hardware-independent Pi0.5 constants and checkpoint schema.

Pi0.5 shares the Pi0 backbone (SigLIP So400m/14 vision, Gemma 2B prefix, Gemma
300M action expert) and the same flow-matching action loop. The checkpoint
differs structurally in three places, all in the action expert:

- state is no longer a continuous suffix token; ``state_proj`` is gone (the
  state is tokenized into the language prefix instead, see ``tokenize``).
- the timestep is injected through a dedicated ``time_mlp_in/out`` whose output
  conditions per-layer AdaRMSNorm, replacing the fused action+time MLP.
- every action-expert norm -- both pre-norms of all 18 layers *and* the final
  norm -- is an AdaRMSNorm with its own modulation Dense
  (``Dense(DECODER_DIM -> 3*DECODER_DIM)`` producing scale/shift/gate).

The language input also changes: because ``max_token_len`` grows to 200 and the
state is tokenized from a runtime value, the checkpoint stores the raw PaliGemma
embedding table rather than precomputed embeddings.

There are two layouts here, and the split is deliberate.

``weight_shapes()`` is the **model contract**: one entry per OpenPI tensor, up
to relayout that is lossless and schedule-independent -- transposes, the q/k/v
concat, the RoPE permutation, and absorbing a plain per-channel RMSNorm scale
into the single GEMM that consumes it. Read it against OpenPI's state dict.

``runtime_shapes(steps)`` is what an execution target actually loads. The
AdaRMSNorm condition is a pure function of the flow-matching timestep, and the
timestep schedule is fixed at capture, so all 37 modulation Dense layers resolve
to constant per-step vectors: 116 M parameters that never have to be streamed.
``weights.fold`` performs that transformation. The line between the two is
exactly "does it depend on the inference schedule" -- everything that does stays
out of the model contract.
"""

# --- Shared backbone (identical to Pi0) ------------------------------------

VISION_LAYERS = 27
VISION_DIM = 1152
VISION_FFN = 4304
VISION_TOKENS = 256  # 224 / 14 == 16 patches per side

ENCODER_LAYERS = 18
ENCODER_DIM = 2048
ENCODER_FFN = 16384

HEAD_DIM = 256
DECODER_HEADS = 8
ROPE_THETA = 10000

# Multiprojection QKV width shared by both Gemma experts:
#   Q (DECODER_HEADS * HEAD_DIM = 2048) + K (HEAD_DIM) + V (HEAD_DIM) = 2560.
QKV_WIDTH = 2560

# --- Action expert (Gemma 300M) --------------------------------------------

DECODER_DIM = 1024
DECODER_FFN = 4096
ACTION_DIM = 32

# --- Pi0.5-specific ---------------------------------------------------------

MAX_TOKEN_LEN = 200
PALIGEMMA_VOCAB_SIZE = 257_152
ADARMS_MOD_DIM = 3 * DECODER_DIM  # scale, shift, gate

#: Flow-matching steps OpenPI samples with by default (`sample_actions`), and
#: therefore the number of distinct AdaRMSNorm conditions a folded checkpoint
#: has to carry.
DEFAULT_FLOW_STEPS = 10

#: Sine-cosine timestep embedding range (`models/pi0.py:161`).
TIME_MIN_PERIOD = 4e-3
TIME_MAX_PERIOD = 4.0


def weight_shapes() -> dict[str, tuple[int, ...]]:
    """Expected checkpoint tensors for a Pi0.5 model, aligned with OpenPI.

    Unlike Pi0, there is no prompt-length argument: the vocabulary table size is
    fixed and the prompt length only affects runtime buffers, not weights.
    """
    return {
        # --- vision: identical to Pi0 ---
        "vision_patch_embedding_w": (14, 14, 3, VISION_DIM),
        "vision_patch_embedding_b": (VISION_DIM,),
        "vision_position_embedding": (VISION_TOKENS, VISION_DIM),
        "vision_attn_qkv_w": (VISION_LAYERS, VISION_DIM, 3 * VISION_DIM),
        "vision_attn_qkv_b": (VISION_LAYERS, 3 * VISION_DIM),
        "vision_attn_o_w": (VISION_LAYERS, VISION_DIM, VISION_DIM),
        "vision_attn_o_b": (VISION_LAYERS, VISION_DIM),
        "vision_ffn_up_w": (VISION_LAYERS, VISION_DIM, VISION_FFN),
        "vision_ffn_up_b": (VISION_LAYERS, VISION_FFN),
        "vision_ffn_down_w": (VISION_LAYERS, VISION_FFN, VISION_DIM),
        "vision_ffn_down_b": (VISION_LAYERS, VISION_DIM),
        "vision_pre_attn_norm_w": (VISION_LAYERS, VISION_DIM),
        "vision_pre_attn_norm_b": (VISION_LAYERS, VISION_DIM),
        "vision_pre_ffn_norm_w": (VISION_LAYERS, VISION_DIM),
        "vision_pre_ffn_norm_b": (VISION_LAYERS, VISION_DIM),
        "vision_final_norm_w": (VISION_DIM,),
        "vision_final_norm_b": (VISION_DIM,),

        # --- encoder (Gemma 2B prefix): identical to Pi0 ---
        # Its RMSNorm scales are plain and schedule-independent, so they are
        # absorbed into the qkv / gate / up weights exactly as in Pi0.
        "encoder_multi_modal_projector_w": (VISION_DIM, ENCODER_DIM),
        "encoder_multi_modal_projector_b": (ENCODER_DIM,),
        "encoder_attn_qkv_w": (ENCODER_LAYERS, ENCODER_DIM, QKV_WIDTH),
        "encoder_attn_o_w": (ENCODER_LAYERS, ENCODER_DIM, ENCODER_DIM),
        "encoder_ffn_gate_w": (ENCODER_LAYERS, ENCODER_DIM, ENCODER_FFN),
        "encoder_ffn_up_w": (ENCODER_LAYERS, ENCODER_DIM, ENCODER_FFN),
        "encoder_ffn_down_w": (ENCODER_LAYERS, ENCODER_FFN, ENCODER_DIM),

        # --- decoder (action expert): the Pi0.5 differences live here -------
        # Gone vs Pi0: decoder_state_in_proj_{w,b} (state moved to prefix),
        # decoder_action_fused_time_biases and decoder_action_mlp_{w,b} (the
        # action+time concat MLP is replaced by time_mlp + AdaRMSNorm).
        "decoder_action_in_proj_w": (ACTION_DIM, DECODER_DIM),
        "decoder_action_in_proj_b": (DECODER_DIM,),
        "decoder_time_mlp_in_w": (DECODER_DIM, DECODER_DIM),
        "decoder_time_mlp_in_b": (DECODER_DIM,),
        "decoder_time_mlp_out_w": (DECODER_DIM, DECODER_DIM),
        "decoder_time_mlp_out_b": (DECODER_DIM,),
        # AdaRMSNorm modulation Dense: one before attention and one before the
        # FFN in every layer, plus one on the expert's final norm (whose gate
        # output is discarded). 37 sites, 116 M parameters.
        "decoder_ada_rms_attn_w": (ENCODER_LAYERS, DECODER_DIM, ADARMS_MOD_DIM),
        "decoder_ada_rms_attn_b": (ENCODER_LAYERS, ADARMS_MOD_DIM),
        "decoder_ada_rms_ffn_w": (ENCODER_LAYERS, DECODER_DIM, ADARMS_MOD_DIM),
        "decoder_ada_rms_ffn_b": (ENCODER_LAYERS, ADARMS_MOD_DIM),
        "decoder_ada_rms_final_w": (DECODER_DIM, ADARMS_MOD_DIM),
        "decoder_ada_rms_final_b": (ADARMS_MOD_DIM,),
        # MQA attention and FFN, identical to Pi0.
        "decoder_attn_qkv_w": (ENCODER_LAYERS, DECODER_DIM, QKV_WIDTH),
        "decoder_attn_o_w": (ENCODER_LAYERS, DECODER_HEADS * HEAD_DIM, DECODER_DIM),
        "decoder_ffn_gate_w": (ENCODER_LAYERS, DECODER_DIM, DECODER_FFN),
        "decoder_ffn_up_w": (ENCODER_LAYERS, DECODER_DIM, DECODER_FFN),
        "decoder_ffn_down_w": (ENCODER_LAYERS, DECODER_FFN, DECODER_DIM),
        "decoder_action_out_proj_w": (DECODER_DIM, ACTION_DIM),
        "decoder_action_out_proj_b": (ACTION_DIM,),

        # --- language: raw embedding table (Pi0 stored precomputed embeds) ---
        "vocab_embeddings": (PALIGEMMA_VOCAB_SIZE, ENCODER_DIM),
    }


#: Tensors that `weights.fold` consumes and that no longer exist at runtime.
FOLDED_AWAY = (
    "decoder_time_mlp_in_w", "decoder_time_mlp_in_b",
    "decoder_time_mlp_out_w", "decoder_time_mlp_out_b",
    "decoder_ada_rms_attn_w", "decoder_ada_rms_attn_b",
    "decoder_ada_rms_ffn_w", "decoder_ada_rms_ffn_b",
    "decoder_ada_rms_final_w", "decoder_ada_rms_final_b",
)


def runtime_shapes(steps: int = DEFAULT_FLOW_STEPS) -> dict[str, tuple[int, ...]]:
    """Tensors an execution target loads, with the AdaRMSNorm condition resolved.

    Everything outside the action expert is untouched. Inside it, the 37
    modulation Dense layers and the time MLP collapse into per-step vectors:

    ``x_hat @ W`` with ``x_hat = rms(x) * (1 + scale) + shift`` becomes
    ``rstd(x) * ((x * ada_scale) @ W) + shift_bias``, so a consuming GEMM needs
    a per-K-column scale on its A operand and a per-N constant bias, and the
    gated residual needs a per-N gate vector. The final norm goes further: its
    scale, its shift and the Euler ``dt`` all fold into the output projection,
    which keeps Pi0's ``out += bias + rms(x) @ weight`` call shape exactly.
    """
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")

    shapes = weight_shapes()
    for name in FOLDED_AWAY:
        del shapes[name]
    shapes.update({
        # (1 + scale) and gate, per step and layer.
        "decoder_ada_attn_scale": (steps, ENCODER_LAYERS, DECODER_DIM),
        "decoder_ada_attn_gate": (steps, ENCODER_LAYERS, DECODER_DIM),
        "decoder_ada_ffn_scale": (steps, ENCODER_LAYERS, DECODER_DIM),
        "decoder_ada_ffn_gate": (steps, ENCODER_LAYERS, DECODER_DIM),
        # shift @ W, precomputed against the weight the kernel actually uses.
        "decoder_qkv_shift_bias": (steps, ENCODER_LAYERS, QKV_WIDTH),
        "decoder_ffn_gate_shift_bias": (steps, ENCODER_LAYERS, DECODER_FFN),
        "decoder_ffn_up_shift_bias": (steps, ENCODER_LAYERS, DECODER_FFN),
        # dt * (1 + scale_final) * W_out  and  dt * (shift_final @ W_out + b_out).
        "decoder_action_out_proj_w": (steps, DECODER_DIM, ACTION_DIM),
        "decoder_action_out_proj_b": (steps, ACTION_DIM),
    })
    return shapes
