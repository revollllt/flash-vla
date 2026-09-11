"""Capture-safe composition of the frozen upstream LingBot modules."""
from __future__ import annotations

import gc
from math import prod
from pathlib import Path
import sys
from types import MethodType

import torch

from flash_vla.models.lingbot.spec import (
    BACKBONE_DIM,
    BACKBONE_FFN,
    BACKBONE_WEIGHT_NAMES,
    CHUNK,
    EXPERT_DIM,
    EXPERT_FFN,
    EXPERT_WEIGHT_NAMES,
    HEAD_DIM,
    KV_HEADS,
    LANGUAGE_SLOTS,
    LAYERS,
    PATCH_ROWS_PER_VIEW,
    PREFIX_LEN,
    QUERY_HEADS,
    VIEWS,
    VISION_DIM,
    VISION_FFN,
    VISION_LAYERS,
    VISION_WEIGHT_NAMES,
    VISUAL_TOKENS_PER_VIEW,
    WEIGHT_NAMES,
)
from flash_vla.runtime.binding import RouteConstraint
from flash_vla.runtime.ops import OpSpec

NAMES = ("lingbot_vision", "lingbot_prefix", "lingbot_action")
WEIGHT_PARAMS = tuple(f"weight_{index:04d}" for index in range(len(WEIGHT_NAMES)))
BACKBONE_WEIGHT_PARAMS = tuple(
    f"backbone_weight_{index:04d}" for index in range(len(BACKBONE_WEIGHT_NAMES))
)
ACTION_WEIGHT_PARAMS = tuple(
    f"action_weight_{index:04d}" for index in range(len(EXPERT_WEIGHT_NAMES))
)

_VISION_PARAMS = tuple(
    param for param, name in zip(WEIGHT_PARAMS, WEIGHT_NAMES)
    if name in set(VISION_WEIGHT_NAMES)
)


def _nbytes(shape, itemsize: int) -> int:
    return prod(shape) * itemsize


def _vision_bytes(shapes, itemsizes):
    read = _nbytes(shapes["pixel_values"], itemsizes["pixel_values"])
    read += sum(_nbytes(shapes[name], itemsizes[name]) for name in _VISION_PARAMS)
    written = _nbytes(shapes["out"], itemsizes["out"])
    return read, written


def _vision_flops(_):
    rows = VIEWS * 256
    linear = 2 * rows * 1176 * VISION_DIM
    per_layer = (
        2 * rows * VISION_DIM * (3 * VISION_DIM)
        + 2 * rows * VISION_DIM * VISION_DIM
        + 6 * rows * VISION_DIM * VISION_FFN
    )
    full_attention = 4 * 4 * VIEWS * 256 * 256 * VISION_DIM
    window_attention = 4 * (VISION_LAYERS - 4) * 12 * 64 * 64 * VISION_DIM
    merger = 2 * 192 * (4 * VISION_DIM) * (4 * VISION_DIM + BACKBONE_DIM)
    return linear + VISION_LAYERS * per_layer + full_attention + window_attention + merger


def _prefix_flops(_):
    rows = PREFIX_LEN
    projections = 2 * rows * BACKBONE_DIM * (
        QUERY_HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM
    )
    attention = 4 * QUERY_HEADS * rows * rows * HEAD_DIM
    output = 2 * rows * QUERY_HEADS * HEAD_DIM * BACKBONE_DIM
    mlp = 6 * rows * BACKBONE_DIM * BACKBONE_FFN
    return LAYERS * (projections + attention + output + mlp)


def _action_flops(_):
    rows = CHUNK + 1
    projections = 2 * rows * EXPERT_DIM * (
        QUERY_HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM
    )
    attention = 4 * QUERY_HEADS * rows * (PREFIX_LEN + rows) * HEAD_DIM
    output = 2 * rows * QUERY_HEADS * HEAD_DIM * EXPERT_DIM
    mlp = 6 * rows * EXPERT_DIM * EXPERT_FFN
    layer = projections + attention + output + mlp
    outer = (
        2 * EXPERT_DIM * 75
        + 2 * CHUNK * 75 * EXPERT_DIM
        + 2 * CHUNK * 2 * EXPERT_DIM * EXPERT_DIM
        + 2 * CHUNK * EXPERT_DIM * 75
    )
    return 10 * (LAYERS * layer + outer)


OPS = (
    OpSpec(
        "lingbot_vision",
        ("pixel_values", "out", "layers") + WEIGHT_PARAMS,
        outputs=("out",),
        weights=WEIGHT_PARAMS,
        flops=_vision_flops,
        bytes_override=_vision_bytes,
    ),
    OpSpec(
        "lingbot_prefix",
        (
            "vision", "image_masks", "language_tokens", "language_masks",
            "prefix_masks", "prefix_k", "prefix_v", "layers",
        ) + BACKBONE_WEIGHT_PARAMS,
        outputs=("prefix_masks", "prefix_k", "prefix_v"),
        weights=BACKBONE_WEIGHT_PARAMS,
        flops=_prefix_flops,
    ),
    OpSpec(
        "lingbot_action",
        (
            "state", "noise", "prefix_masks", "prefix_k", "prefix_v", "actions",
            "velocity_step_0", "steps", "layers",
        ) + ACTION_WEIGHT_PARAMS,
        outputs=("actions", "velocity_step_0"),
        weights=ACTION_WEIGHT_PARAMS,
        flops=_action_flops,
    ),
)

ROUTE_CONSTRAINTS = (
    RouteConstraint.atomic(NAMES, "the initial monolithic stages share one loaded upstream model"),
)


def _fused_vision_attention(visual, scratch, device, rows: int) -> None:
    """Prepare each vision block's q/k/v in one launch instead of about eleven.

    Upstream splits the packed projection into three views flash-attention then
    has to make contiguous, re-chunks and re-casts cos and sin inside every one
    of the 32 blocks for a value that is constant for the whole forward, and
    sends q and k through float32 either side of the rotation. The blocks share
    one set of output buffers because each is consumed by the attention call
    that follows it, before the next block runs on the same stream.
    """
    from lingbotvla.models.vla.pi0 import qwenvl_in_vla as qwen

    from .cuda import pointwise

    heads = visual.blocks[0].attn.num_heads
    head_dim = visual.blocks[0].attn.head_dim if hasattr(visual.blocks[0].attn, "head_dim") \
        else visual.blocks[0].attn.qkv.out_features // (3 * heads)
    buffers = tuple(
        scratch(f"lingbot_vision_{role}", (rows, heads, head_dim), torch.bfloat16, device)
        for role in ("query", "key", "value"))
    tables: dict = {}

    def make_forward(max_seqlen: int):
        def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                    position_embeddings=None):
            packed = self.qkv(hidden_states)
            if not tables:
                # cos/sin are fixed by the window layout for the life of the
                # engine; upstream rebuilds this slice once per block.
                cos, sin = position_embeddings
                half = head_dim // 2
                tables["cos"] = cos[..., :half].float().contiguous()
                tables["sin"] = sin[..., :half].float().contiguous()
            query, key, value = buffers
            pointwise.vision_rope(packed, tables["cos"], tables["sin"], query, key, value)
            out = qwen.flash_attn_varlen_func(
                query, key, value, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
            ).reshape(hidden_states.shape[0], -1)
            return self.proj(out)
        return forward

    full = set(visual.fullatt_block_indexes)
    for index, block in enumerate(visual.blocks):
        block.attn.forward = MethodType(make_forward(256 if index in full else 64), block.attn)


def _patch_vision_attention(visual) -> None:
    from lingbotvla.models.vla.pi0 import qwenvl_in_vla as qwen

    def make_forward(max_seqlen: int):
        def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                    position_embeddings=None):
            seq_length = hidden_states.shape[0]
            q, k, v = self.qkv(hidden_states).reshape(
                seq_length, 3, self.num_heads, -1
            ).permute(1, 0, 2, 3).unbind(0)
            if position_embeddings is None:
                emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                cos, sin = emb.cos(), emb.sin()
            else:
                cos, sin = position_embeddings
            q, k = qwen.apply_rotary_pos_emb_flashatt(
                q.unsqueeze(0), k.unsqueeze(0), cos, sin
            )
            q, k = q.squeeze(0), k.squeeze(0)
            out_fp32 = k.dtype == torch.float32
            if out_fp32:
                q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
            out = qwen.flash_attn_varlen_func(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
            ).reshape(seq_length, -1)
            if out_fp32:
                out = out.float()
            return self.proj(out)
        return forward

    full = set(visual.fullatt_block_indexes)
    for index, block in enumerate(visual.blocks):
        block.attn.forward = MethodType(make_forward(256 if index in full else 64), block.attn)


def _configure_rope_frequency(enabled: bool, cache_rope_tables: bool = False):
    if not enabled:
        from lingbotvla.models.vla.pi0.utils import apply_rope as original_apply_rope
        return original_apply_rope, None

    inverse_timescales = {}
    rotary_tables = {}

    def apply_rope(x, positions, max_wavelength=10_000.0, dtype=torch.float32):
        original_dtype = x.dtype
        width = x.shape[-1]
        half = width // 2
        key = (width, dtype, x.device, max_wavelength)
        inverse_timescale = inverse_timescales.get(key)
        if inverse_timescale is None:
            exponents = (2.0 / width) * torch.arange(
                half, dtype=dtype, device=x.device,
            )
            inverse_timescale = 1.0 / (max_wavelength ** exponents)
            inverse_timescales[key] = inverse_timescale
        table = rotary_tables.get(key) if cache_rope_tables else None
        if table is None:
            radians = torch.einsum(
                "bl,h->blh", positions.to(dtype), inverse_timescale,
            )[..., None, :]
            table = torch.sin(radians), torch.cos(radians)
            if cache_rope_tables:
                rotary_tables[key] = table
        sin, cos = table
        first, second = x.to(dtype).split(half, dim=-1)
        return torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1,
        ).to(original_dtype)

    return apply_rope, rotary_tables.clear


def _linear_patch_embedding(self, hidden_states):
    """Project pre-expanded bf16 patches with the unchanged Conv3d weights."""
    weight = self.proj.weight.flatten(1)
    patches = hidden_states.reshape(-1, weight.shape[1]).to(dtype=weight.dtype)
    return torch.nn.functional.linear(patches, weight)


#: bf16 tensor-core GEMMs want every leading dimension 8 elements aligned. The
#: vision FFN is 3420 wide, which is not, so cuBLAS falls back to the Ampere
#: `cutlass_80_tensorop_bf16_s16816gemm_bf16_256x128_64x3_tn_align2` kernel for
#: all three of its projections: 57.4 us each against a ~10 us compute-bound
#: model and a 5.0 us streaming floor, 5.51 ms of the segment's 10.40 ms
#: (torch-profiler replay, job 614222).
_VISION_FFN_ALIGNMENT = 8


def _pad_vision_ffn(visual, scratch=None, device=None, rows: int = 0) -> None:
    """Pack and zero-pad each vision block's feed-forward to an aligned width.

    The padded lanes carry a zero weight and a zero bias, so they contribute
    `silu(0) * 0 = 0` to the gated product and a zero column to the down
    projection: the value is unchanged, only the K-reduction blocking moves.
    Allocation happens during warmup, before the static arena freezes.
    """
    for block in visual.blocks:
        mlp = block.mlp
        width = mlp.gate_proj.out_features
        padded = -(-width // _VISION_FFN_ALIGNMENT) * _VISION_FFN_ALIGNMENT
        pad = padded - width

        pad_out = torch.nn.functional.pad
        mlp.packed_gate_up_weight = torch.cat(
            [pad_out(mlp.gate_proj.weight, (0, 0, 0, pad)),
             pad_out(mlp.up_proj.weight, (0, 0, 0, pad))], dim=0)
        mlp.packed_gate_up_bias = torch.cat(
            [pad_out(mlp.gate_proj.bias, (0, pad)), pad_out(mlp.up_proj.bias, (0, pad))], dim=0)
        mlp.padded_down_weight = pad_out(mlp.down_proj.weight, (0, pad))
        mlp.padded_width = padded
        mlp.activation = (None if scratch is None else
                          scratch("lingbot_vision_activation", (rows, padded),
                                  torch.bfloat16, device))
        mlp.forward = MethodType(_padded_vision_mlp, mlp)


def _padded_vision_mlp(self, hidden_state):
    """`Qwen2_5_VLMLP` on one packed, width-aligned gated projection.

    With an `activation` workspace bound, the gated product is one launch over
    the packed result instead of torch's separate SiLU and multiply, which
    otherwise pass the 10.5 MB hidden tile three times.
    """
    gate_up = torch.nn.functional.linear(
        hidden_state, self.packed_gate_up_weight, self.packed_gate_up_bias)
    if self.activation is not None and gate_up.shape[0] == self.activation.shape[0]:
        from .cuda import pointwise

        activated = pointwise.silu_multiply(gate_up, self.activation)
    else:
        gate, up = gate_up.split(self.padded_width, dim=-1)
        activated = self.act_fn(gate) * up
    return torch.nn.functional.linear(
        activated, self.padded_down_weight, self.down_proj.bias)


def _fuse_rms_norms(norms, rows: int, width: int, scratch, device) -> None:
    """Bind the single-launch RMSNorm to each of `norms`, over one shared output.

    Every site's result is consumed by the GEMM that follows it before the next
    site runs, and the whole stage replays on one stream, so one buffer per
    (rows, width) is enough. Allocation happens during warmup.
    """
    from .cuda import pointwise

    buffer = scratch(f"lingbot_rms_norm_{rows}x{width}", (rows, width),
                     torch.bfloat16, device)
    for norm in norms:
        def forward(hidden_states, norm=norm):
            if hidden_states.shape != buffer.shape or not hidden_states.is_contiguous():
                return _EAGER_RMS_NORM(norm, hidden_states)
            return pointwise.rms_norm(hidden_states, norm.weight, buffer,
                                      norm.variance_epsilon)

        norm.forward = forward


def _EAGER_RMS_NORM(norm, hidden_states):
    """Upstream's own expression, for a call whose shape the fused buffer misses."""
    values = hidden_states.to(torch.float32)
    variance = values.pow(2).mean(-1, keepdim=True)
    values = values * torch.rsqrt(variance + norm.variance_epsilon)
    return norm.weight * values.to(hidden_states.dtype)


def _build_policy(weight_values, layers: int, cache_rope_frequency: bool, assets,
                  linear_patch_embedding: bool = False, cache_rope_tables: bool = False,
                  pack_expert_projections: bool = False, grouped_attention: bool = False,
                  pad_vision_ffn: bool = False, fused_vision_norm: bool = False,
                  fused_vision_attention: bool = False, scratch=None):
    import yaml
    from lerobot.configs.policies import PreTrainedConfig
    from transformers import AutoConfig

    upstream, checkpoint, qwen_path = (Path(assets[role]) for role in ("upstream", "checkpoint", "qwen"))
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from deploy.lingbot_vla_policy import LingBotVlaInferencePolicy, merge_qwen_config

    config = PreTrainedConfig.from_pretrained(checkpoint)
    with (checkpoint / "lingbotvla_cli.yaml").open() as source:
        training = yaml.safe_load(source)
    values = {**training["model"], **training["train"]}
    config.__dict__.update({key: value for key, value in values.items()
                            if not hasattr(config, key)})
    config.attention_implementation = "eager"
    config.tokenizer_path = str(qwen_path)
    config = merge_qwen_config(config, AutoConfig.from_pretrained(qwen_path))
    if training["model"].get("vocab_size", 0):
        config.vocab_size = training["model"]["vocab_size"]
    config.use_cache = True

    policy = LingBotVlaInferencePolicy(config, tokenizer_path=str(qwen_path))
    state = dict(zip(WEIGHT_NAMES, weight_values))
    policy.load_state_dict(state, strict=True, assign=True)
    device = weight_values[0].device
    policy.to(device=device, dtype=torch.bfloat16).eval()
    core = policy.model
    core.qwenvl_with_expert.qwenvl.config.num_hidden_layers = layers
    visual = core.qwenvl_with_expert.qwenvl.visual
    if fused_vision_attention:
        _fused_vision_attention(visual, scratch, device, VIEWS * PATCH_ROWS_PER_VIEW)
    else:
        _patch_vision_attention(visual)
    if linear_patch_embedding:
        visual.patch_embed.forward = MethodType(_linear_patch_embedding, visual.patch_embed)
    if pad_vision_ffn:
        _pad_vision_ffn(visual, scratch if fused_vision_norm else None, device,
                        VIEWS * PATCH_ROWS_PER_VIEW)
    if fused_vision_norm:
        norms = [norm for block in visual.blocks for norm in (block.norm1, block.norm2)]
        _fuse_rms_norms(norms + [visual.merger.ln_q], VIEWS * PATCH_ROWS_PER_VIEW,
                        VISION_DIM, scratch, device)
    from lingbotvla.models.vla.pi0 import modeling_lingbot_vla as lingbot
    apply_rope, clear_rope_tables = _configure_rope_frequency(cache_rope_frequency, cache_rope_tables)
    original_forward = core.qwenvl_with_expert.forward

    def forward_with_rope(*args, **kwargs):
        # Upstream resolves RoPE through a module global. Bind this policy's
        # function while constructing its graph, including later recaptures.
        previous = lingbot.apply_rope
        lingbot.apply_rope = apply_rope
        try:
            if cache_rope_tables:
                clear_rope_tables()
            return original_forward(*args, **kwargs)
        finally:
            lingbot.apply_rope = previous

    core.qwenvl_with_expert.forward = forward_with_rope
    if pack_expert_projections:
        _pack_expert_projections(core, layers)
    if grouped_attention:
        core.qwenvl_with_expert.attention_interface = _grouped_attention
    gc.collect()
    return core


def _grouped_attention(query_states, key_states, value_states, attention_mask):
    """Eager attention that folds the GQA groups into the batched-matmul rows.

    Upstream materializes `repeat`ed key/value copies, one per query head, so
    each expert layer-step writes and re-reads 16/2 times the cache it needs.
    Folding the group axis into the row axis of the same batched matmul leaves
    every dot product and its K reduction unchanged.

    Shapes: query `[b, lq, H, D]`, key/value `[b, lk, Hkv, D]`, mask
    `[b, lq, lk]`, all on CUDA; returns `[b, lq, H * D]`. Capture-safe.
    """
    batch, q_len, heads, head_dim = query_states.shape
    kv_heads = key_states.shape[2]
    group = heads // kv_heads
    queries = query_states.permute(0, 2, 1, 3).reshape(batch, kv_heads, group * q_len, head_dim)
    keys = key_states.permute(0, 2, 1, 3)
    values = value_states.permute(0, 2, 1, 3)

    weights = torch.matmul(queries, keys.transpose(-1, -2))
    weights = weights.view(batch, heads, q_len, -1)
    weights *= head_dim ** -0.5
    weights = torch.where(attention_mask[:, None, :, :], weights, -2.3819763e38)
    probs = torch.nn.functional.softmax(weights, dim=-1).to(dtype=value_states.dtype)

    output = torch.matmul(probs.view(batch, kv_heads, group * q_len, -1), values)
    output = output.view(batch, kv_heads, group, q_len, head_dim)
    return output.permute(0, 3, 1, 2, 4).reshape(batch, q_len, heads * head_dim)


def _expert_layer_forward(self, hidden_states, att_output=None, start=0, end=0,
                          compute_kqv=False, norm_qkv=False, output_atten=False,
                          ada_cond=None, gate=None, **kwargs):
    """One expert decoder layer on packed projections, replacing the upstream split.

    Two GEMMs replace five: q/k/v share one packed weight and gate/up share
    another, concatenated along the output dimension so every dot product and
    its K reduction are the ones upstream computes. The single float32 cast of
    the packed projection replaces the three the caller would otherwise make,
    and the residual is kept rather than cloned, because the AdaRMS that
    follows returns a new tensor and never writes through its input.

    `hidden_states` is `[b, l, EXPERT_DIM]` bf16 on CUDA. Capture-safe: the
    packed weights are bound by `_pack_expert_projections` during warmup.
    """
    if compute_kqv:
        normed = self.input_layernorm(hidden_states, ada_cond)
        packed = torch.nn.functional.linear(
            normed, self.packed_qkv_weight, self.packed_qkv_bias).float()
        query, key, value = packed.split(self.qkv_widths, dim=-1)
        head_dim = self.self_attn.head_dim
        return (query.view(*query.shape[:-1], -1, head_dim),
                key.view(*key.shape[:-1], -1, head_dim),
                value.view(*value.shape[:-1], -1, head_dim), None)
    if not output_atten:
        raise ValueError("the LingBot expert layer runs either projections or the attention tail")

    weight_dtype = self.self_attn.o_proj.weight.dtype
    out = self.self_attn.o_proj(att_output[:, start:end].to(weight_dtype))
    out += hidden_states
    normed = self.post_attention_layernorm(out, ada_cond)
    gate_up = torch.nn.functional.linear(normed, self.packed_gate_up_weight)
    gate_values, up_values = gate_up.split(self.gate_up_widths, dim=-1)
    out += self.mlp.down_proj(torch.nn.functional.silu(gate_values) * up_values)
    return out


def _pack_expert_projections(core, depth: int) -> None:
    """Bind each expert layer's packed q/k/v and gate/up weights and its leaner forward.

    Allocation happens here, while the runner is still warming up, because the
    static arena refuses to allocate once capture has begun.
    """
    layers = core.qwenvl_with_expert.qwen_expert.model.layers
    for layer in layers[:depth]:
        attention = layer.self_attn
        layer.packed_qkv_weight = torch.cat(
            [attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight], dim=0)
        layer.packed_qkv_bias = torch.cat(
            [attention.q_proj.bias, attention.k_proj.bias, attention.v_proj.bias], dim=0)
        layer.qkv_widths = [attention.q_proj.out_features, attention.k_proj.out_features,
                            attention.v_proj.out_features]
        layer.packed_gate_up_weight = torch.cat(
            [layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight], dim=0)
        layer.gate_up_widths = [layer.mlp.gate_proj.out_features,
                                layer.mlp.up_proj.out_features]
        layer.forward = MethodType(_expert_layer_forward, layer)


def _prepare_time_modulation(core, steps, dtype, device, *, fuse_norm=False):
    # Pi0.5 precomputes fixed-timestep AdaRMS conditions too. Preserve LingBot's
    # own BF16 schedule, embedding, and linear arithmetic rather than its fold.
    from lingbotvla.models.vla.pi0.utils import create_sinusoidal_pos_embedding

    dt = torch.tensor(-1.0 / steps, dtype=dtype, device=device)
    time = torch.tensor(1.0, dtype=dtype, device=device)
    conditions = []
    for _ in range(steps):
        condition = create_sinusoidal_pos_embedding(
            time.expand(1), core.config.proj_width, 4e-3, 4.0, device=device,
        ).to(dtype=dtype)
        if core.config.separate_time_proj:
            condition = torch.nn.functional.silu(core.time_mlp_in(condition))
            condition = torch.nn.functional.silu(core.time_mlp_out(condition))
        conditions.append(condition)
        time += dt

    step = [0]
    modulation = (step, conditions)
    layers = core.qwenvl_with_expert.qwen_expert.model.layers
    depth = core.qwenvl_with_expert.qwenvl.config.num_hidden_layers
    for layer in layers[:depth]:
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            for projection in (norm.gamma, norm.beta):
                values = [projection(condition) for condition in conditions]

                def cached_forward(condition, values=values):
                    return values[step[0]]

                projection.forward = cached_forward
            if fuse_norm:
                from .fused_norm import ada_rms

                def normalized(hidden_states, condition, norm=norm):
                    return ada_rms(hidden_states, norm.weight,
                                   norm.gamma(condition).unsqueeze(1),
                                   norm.beta(condition).unsqueeze(1),
                                   norm.variance_epsilon)

                norm.forward = normalized
    return modulation


class _State:
    def __init__(self, cache_rope_frequency: bool, assets, linear_patch_embedding: bool,
                 cache_rope_tables: bool, precompute_time_modulation: bool, fuse_norm: bool,
                 pack_expert_projections: bool = False, grouped_attention: bool = False,
                 specialized_loop: bool = False, fused_rope: bool = False,
                 fused_attention: bool = False, specialized_prefix: bool = False,
                 pad_vision_ffn: bool = False, fused_vision_norm: bool = False,
                 fused_vision_attention: bool = False,
                 fused_mlp: bool = False, fused_prefix_pointwise: bool = False,
                 attention_kernel: bool = False, skinny_gemm: bool = False,
                 split_attention: bool = False, fused_gate: bool = False,
                 scratch=None) -> None:
        self.core = None
        self.loop = None
        self.specialized_loop = specialized_loop
        self.fused_rope = fused_rope
        self.fused_attention = fused_attention
        self.specialized_prefix = specialized_prefix
        self.pad_vision_ffn = pad_vision_ffn
        self.fused_vision_norm = fused_vision_norm
        self.fused_vision_attention = fused_vision_attention
        self.fused_mlp = fused_mlp
        self.fused_prefix_pointwise = fused_prefix_pointwise
        self.attention_kernel = attention_kernel
        self.skinny_gemm = skinny_gemm
        self.split_attention = split_attention
        self.fused_gate = fused_gate
        self.scratch = scratch
        self.prefix_pass = None
        self.vision_metadata = None
        self.action_constants = None
        self.precompute_time_modulation = precompute_time_modulation
        self.fuse_norm = fuse_norm
        self.time_modulation_step = None
        self.cache_rope_frequency = cache_rope_frequency
        self.linear_patch_embedding = linear_patch_embedding
        self.cache_rope_tables = cache_rope_tables
        self.pack_expert_projections = pack_expert_projections
        self.grouped_attention = grouped_attention
        self.assets = assets

    def ensure(self, weights, layers: int):
        if self.core is None:
            self.core = _build_policy(weights, layers, self.cache_rope_frequency, self.assets,
                                      linear_patch_embedding=self.linear_patch_embedding,
                                      cache_rope_tables=self.cache_rope_tables,
                                      pack_expert_projections=self.pack_expert_projections,
                                      grouped_attention=self.grouped_attention,
                                      pad_vision_ffn=self.pad_vision_ffn,
                                      fused_vision_norm=self.fused_vision_norm,
                                      fused_vision_attention=self.fused_vision_attention,
                                      scratch=self.scratch)
        return self.core


def make_wrappers(scratch, selected_names=None, *, cache_rope_frequency=False,
                  linear_patch_embedding=False, cache_rope_tables=False,
                  precompute_time_modulation=False, fuse_norm=False,
                  pack_expert_projections=False, grouped_attention=False,
                  specialized_loop=False, fused_rope=False, fused_attention=False,
                  specialized_prefix=False, pad_vision_ffn=False, fused_vision_norm=False,
                  fused_mlp=False, fused_prefix_pointwise=False, attention_kernel=False,
                  skinny_gemm=False, split_attention=False, fused_vision_attention=False,
                  fused_gate=False):
    state = _State(cache_rope_frequency, scratch.assets, linear_patch_embedding,
                   cache_rope_tables, precompute_time_modulation, fuse_norm,
                   pack_expert_projections=pack_expert_projections,
                   grouped_attention=grouped_attention,
                   specialized_loop=specialized_loop, fused_rope=fused_rope,
                   fused_attention=fused_attention,
                   specialized_prefix=specialized_prefix, pad_vision_ffn=pad_vision_ffn,
                   fused_vision_norm=fused_vision_norm,
                   fused_vision_attention=fused_vision_attention, fused_mlp=fused_mlp,
                   fused_prefix_pointwise=fused_prefix_pointwise,
                   attention_kernel=attention_kernel, skinny_gemm=skinny_gemm,
                   split_attention=split_attention, fused_gate=fused_gate,
                   scratch=scratch)

    @torch.no_grad()
    def vision(pixel_values, out, layers, *weights):
        core = state.ensure(weights, layers)
        module = core.qwenvl_with_expert
        visual = module.qwenvl.visual
        if state.vision_metadata is None:
            grid = torch.tensor([[1, 16, 16]] * VIEWS, device=pixel_values.device)
            rotary, window, cu_window, cu_full = visual.preprcess_grid_thw(grid)
            state.vision_metadata = (
                rotary, window.to(pixel_values.device), cu_window, cu_full,
            )
        rotary, window, cu_window, cu_full = state.vision_metadata
        embeds = visual(
            pixel_values,
            grid_thw=None,
            rotary_pos_emb=rotary,
            window_index=window,
            cu_window_seqlens=cu_window,
            cu_seqlens=cu_full,
        )
        out.copy_(embeds.reshape(VIEWS, VISUAL_TOKENS_PER_VIEW, BACKBONE_DIM))

    @torch.no_grad()
    def prefix(vision, image_masks, language_tokens, language_masks,
               prefix_masks, prefix_k, prefix_v, layers, *weights):
        from lingbotvla.models.vla.pi0.utils import make_att_2d_masks

        core = state.core
        if state.specialized_prefix:
            if state.prefix_pass is None:
                from .prefix_pass import PrefixPass

                state.prefix_pass = PrefixPass(core, layers=layers, scratch=scratch,
                                               device=vision.device,
                                               fused_pointwise=state.fused_prefix_pointwise,
                                               attention_kernel=state.attention_kernel)
            state.prefix_pass.run(vision, image_masks, language_tokens, language_masks,
                                  prefix_masks, prefix_k, prefix_v)
            return
        module = core.qwenvl_with_expert
        image_prefix = vision.reshape(1, -1, vision.shape[-1])
        expanded_image_masks = image_masks.unsqueeze(0).repeat_interleave(
            VISUAL_TOKENS_PER_VIEW, dim=1
        )
        language = module.embed_language_tokens(language_tokens)
        embeddings = torch.cat((image_prefix, language), dim=1)
        masks = torch.cat((expanded_image_masks, language_masks), dim=1)
        prefix_masks.copy_(masks)
        attention = make_att_2d_masks(masks, torch.zeros_like(masks))
        positions = torch.cumsum(masks, dim=1) - 1
        _, cache = module.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[embeddings, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        for layer in range(layers):
            prefix_k[layer].copy_(cache[layer]["key_states"][0])
            prefix_v[layer].copy_(cache[layer]["value_states"][0])

    @torch.no_grad()
    def action(state_tensor, noise, prefix_masks, prefix_k, prefix_v, actions,
               velocity_step_0, steps, layers, *weights):
        core = state.core
        cache = {
            layer: {
                "key_states": prefix_k[layer].unsqueeze(0),
                "value_states": prefix_v[layer].unsqueeze(0),
            }
            for layer in range(layers)
        }
        if state.action_constants is None:
            state.action_constants = (
                torch.tensor(-1.0 / steps, dtype=noise.dtype, device=noise.device),
                torch.tensor(1.0, dtype=noise.dtype, device=noise.device),
            )
        if state.precompute_time_modulation and state.time_modulation_step is None:
            state.time_modulation_step, conditions = _prepare_time_modulation(
                core, steps, noise.dtype, noise.device, fuse_norm=state.fuse_norm,
            )
            if state.specialized_loop:
                from .expert_loop import ExpertLoop

                state.loop = ExpertLoop(
                    core, layers=layers, steps=steps, conditions=conditions,
                    time_step=state.time_modulation_step, scratch=scratch,
                    device=noise.device, dtype=noise.dtype,
                    fused_rope=state.fused_rope,
                    fused_attention=state.fused_attention,
                    fused_mlp=state.fused_mlp,
                    attention_kernel=state.attention_kernel,
                    skinny_gemm=state.skinny_gemm,
                    split_attention=state.split_attention,
                    fused_gate=state.fused_gate,
                )
        if state.loop is not None:
            state.loop.run(state_tensor, noise, prefix_masks, prefix_k, prefix_v,
                           actions, velocity_step_0)
            return
        dt, initial_time = state.action_constants
        time = initial_time.clone()
        current = noise.clone()
        for step in range(steps):
            if state.time_modulation_step is not None:
                state.time_modulation_step[0] = step
            velocity = core.predict_velocity(
                state_tensor, prefix_masks, cache, current, time.expand(1)
            )
            if step == 0:
                velocity_step_0.copy_(velocity)
            current += dt * velocity
            time += dt
        actions.copy_(current)

    wrappers = {
        "lingbot_vision": vision,
        "lingbot_prefix": prefix,
        "lingbot_action": action,
    }
    return {name: wrappers[name] for name in (selected_names or NAMES)}


__all__ = [
    "ACTION_WEIGHT_PARAMS", "BACKBONE_WEIGHT_PARAMS", "NAMES", "OPS",
    "ROUTE_CONSTRAINTS", "WEIGHT_PARAMS", "make_wrappers",
]
