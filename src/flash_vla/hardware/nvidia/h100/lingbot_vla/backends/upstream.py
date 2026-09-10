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


def _build_policy(weight_values, layers: int, cache_rope_frequency: bool, assets,
                  linear_patch_embedding: bool = False, cache_rope_tables: bool = False):
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
    _patch_vision_attention(visual)
    if linear_patch_embedding:
        visual.patch_embed.forward = MethodType(_linear_patch_embedding, visual.patch_embed)
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
    gc.collect()
    return core


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
    return step


class _State:
    def __init__(self, cache_rope_frequency: bool, assets, linear_patch_embedding: bool,
                 cache_rope_tables: bool, precompute_time_modulation: bool, fuse_norm: bool) -> None:
        self.core = None
        self.vision_metadata = None
        self.action_constants = None
        self.precompute_time_modulation = precompute_time_modulation
        self.fuse_norm = fuse_norm
        self.time_modulation_step = None
        self.cache_rope_frequency = cache_rope_frequency
        self.linear_patch_embedding = linear_patch_embedding
        self.cache_rope_tables = cache_rope_tables
        self.assets = assets

    def ensure(self, weights, layers: int):
        if self.core is None:
            self.core = _build_policy(weights, layers, self.cache_rope_frequency, self.assets,
                                      linear_patch_embedding=self.linear_patch_embedding,
                                      cache_rope_tables=self.cache_rope_tables)
        return self.core


def make_wrappers(scratch, selected_names=None, *, cache_rope_frequency=False,
                  linear_patch_embedding=False, cache_rope_tables=False,
                  precompute_time_modulation=False, fuse_norm=False):
    state = _State(cache_rope_frequency, scratch.assets, linear_patch_embedding,
                   cache_rope_tables, precompute_time_modulation, fuse_norm)

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
            state.time_modulation_step = _prepare_time_modulation(
                core, steps, noise.dtype, noise.device, fuse_norm=state.fuse_norm,
            )
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
