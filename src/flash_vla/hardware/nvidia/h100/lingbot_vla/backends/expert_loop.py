"""LingBot's denoising loop with its per-forward invariants lifted out of it.

Upstream reaches the expert stack through the shared `QwenvlWithExpertModel`
forward, which is written for either tower and for both the prefix fill and the
denoising read. On this Target the denoising read is the only case: batch one,
264 prefix tokens, 51 suffix tokens, ten Euler steps over 36 layers. Under that
shape the shared path rebuilds, once per layer per step, a suffix mask, a RoPE
table, three single-element concatenations and a fresh copy of the entire
prefix key/value cache -- 360 repetitions of work whose inputs never change
inside a forward.

This module runs the same arithmetic with those invariants computed once and
with the key/value cache resident, so the step loop writes only the 51 suffix
rows it actually recomputes. Two optional stages replace the remaining
pointwise chains with the hand-written kernels in `cuda/`.
"""
from __future__ import annotations

import torch

from flash_vla.models.lingbot.spec import (
    CHUNK,
    HEAD_DIM,
    KV_HEADS,
    PREFIX_LEN,
    SUFFIX_LEN,
)

_ROPE_WAVELENGTH = 10_000.0
_CACHE_LEN = PREFIX_LEN + SUFFIX_LEN
_SCALE = HEAD_DIM ** -0.5


def _apply_rope(values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Upstream's half-split rotation on `[b, l, h, HEAD_DIM]` float32 values."""
    first, second = values.split(HEAD_DIM // 2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


def _grouped_attention(query, key, value, mask):
    """Eager attention that folds the GQA groups into the batched-matmul rows.

    Upstream materializes a `repeat`ed key and value, one copy per query head,
    so every layer-step writes and re-reads 8x the cache it needs. Folding the
    group axis into the row axis of the same batched matmul leaves each dot
    product and its K reduction unchanged.

    `query` is `[b, lq, H, D]`, `key`/`value` are `[b, lk, Hkv, D]` and `mask`
    is `[b, lq, lk]`, all float32 on CUDA. Returns `[b, lq, H * D]`.
    """
    batch, q_len, heads, head_dim = query.shape
    kv_heads = key.shape[2]
    group = heads // kv_heads
    queries = query.permute(0, 2, 1, 3).reshape(batch, kv_heads, group * q_len, head_dim)
    keys = key.permute(0, 2, 1, 3)
    values = value.permute(0, 2, 1, 3)

    weights = torch.matmul(queries, keys.transpose(-1, -2)).view(batch, heads, q_len, -1)
    weights *= head_dim ** -0.5
    weights = torch.where(mask[:, None, :, :], weights, -2.3819763e38)
    probs = torch.nn.functional.softmax(weights, dim=-1).to(dtype=value.dtype)

    output = torch.matmul(probs.view(batch, kv_heads, group * q_len, -1), values)
    output = output.view(batch, kv_heads, group, q_len, head_dim)
    return output.permute(0, 3, 1, 2, 4).reshape(batch, q_len, heads * head_dim)


class ExpertLoop:
    """The ten Euler steps of one LingBot forward over a resident key/value cache.

    Built once, during runner warmup, because the static arena refuses to
    allocate after capture starts. `run` is capture-safe: it writes only into
    buffers bound here and into the graph's own output buffers.

    `fused_rope` replaces the projection epilogue with one CUDA launch;
    `fused_attention` additionally keeps the query and the cache head-major, so
    the attention matmuls need no transposing copies, and replaces the score
    pointwise chain and the output transpose with two more launches. The second
    stage implies the first, because the head-major layout is produced by that
    same projection kernel.
    """

    def __init__(self, core, *, layers: int, steps: int, conditions, time_step,
                 scratch, device, dtype, fused_rope: bool = False,
                 fused_attention: bool = False, fused_mlp: bool = False,
                 attention_kernel: bool = False) -> None:
        self.core = core
        self.steps = steps
        self.depth = layers
        self.conditions = conditions
        self.time_step = time_step
        self.fused_rope = fused_rope or fused_attention
        self.fused_attention = fused_attention
        self.fused_mlp = fused_mlp
        self.attention_kernel = attention_kernel

        expert = core.qwenvl_with_expert.qwen_expert.model
        self.layers = tuple(expert.layers[:layers])
        self.final_norm = expert.norm
        query_heads = self.layers[0].qkv_widths[0] // HEAD_DIM
        self.group = query_heads // KV_HEADS

        cache_shape = ((layers, KV_HEADS, _CACHE_LEN, HEAD_DIM) if fused_attention
                       else (layers, _CACHE_LEN, KV_HEADS, HEAD_DIM))
        self.key_cache = scratch("lingbot_expert_key_cache", cache_shape, torch.float32, device)
        self.value_cache = scratch("lingbot_expert_value_cache", cache_shape,
                                   torch.float32, device)
        query_shape = ((query_heads, SUFFIX_LEN, HEAD_DIM) if fused_attention
                       else (SUFFIX_LEN, query_heads, HEAD_DIM))
        self.query = scratch("lingbot_expert_query", query_shape, torch.float32, device)
        self.attention_out = scratch("lingbot_expert_attention",
                                     (SUFFIX_LEN, query_heads * HEAD_DIM),
                                     torch.bfloat16, device)
        if fused_mlp:
            width = self.layers[0].self_attn.o_proj.out_features
            for role in ("hidden", "residual", "normed"):
                setattr(self, role, scratch(f"lingbot_expert_{role}", (SUFFIX_LEN, width),
                                            torch.bfloat16, device))
            self.activation = scratch("lingbot_expert_activation",
                                      (SUFFIX_LEN, self.layers[0].gate_up_widths[0]),
                                      torch.bfloat16, device)
            # The first AdaRMS of a step has no residual yet; a zero one keeps
            # it on the same kernel, since bf16(x + 0) is exactly x.
            self.no_residual = scratch("lingbot_expert_zero", (SUFFIX_LEN, width),
                                       torch.bfloat16, device)
            self.no_residual.zero_()

        self.suffix_pad = torch.ones((1, SUFFIX_LEN), dtype=torch.bool, device=device)
        # Upstream's suffix mask_ar: the state token is its own block and the
        # fifty action tokens share one bidirectional block after it.
        self.suffix_att = torch.zeros((1, SUFFIX_LEN), dtype=torch.bool, device=device)
        self.suffix_att[:, :2] = True

        half = HEAD_DIM // 2
        exponents = (2.0 / HEAD_DIM) * torch.arange(half, dtype=torch.float32, device=device)
        self.inverse_timescale = 1.0 / (_ROPE_WAVELENGTH ** exponents)
        self.dt = torch.tensor(-1.0 / steps, dtype=dtype, device=device)
        if self.fused_rope:
            self._bind_kernels(device, half)

    def _bind_kernels(self, device, half: int) -> None:
        """Load the CUDA library and exercise it once, before capture begins."""
        from .cuda import pointwise

        self._kernels = pointwise
        pointwise.rope_project(
            torch.zeros((SUFFIX_LEN, sum(self.layers[0].qkv_widths)),
                        dtype=torch.bfloat16, device=device),
            torch.zeros((SUFFIX_LEN, half), dtype=torch.float32, device=device),
            torch.zeros((SUFFIX_LEN, half), dtype=torch.float32, device=device),
            *self._slots(0))
        if self.fused_attention:
            pointwise.masked_softmax(
                torch.zeros((1, SUFFIX_LEN, _CACHE_LEN), dtype=torch.float32, device=device),
                torch.ones((SUFFIX_LEN, _CACHE_LEN), dtype=torch.bool, device=device), _SCALE)
            pointwise.attention_epilogue(self.query, self.attention_out)
            if self.attention_kernel:
                pointwise.fused_attention(
                    self.query, self.key_cache[0], self.value_cache[0],
                    torch.ones((SUFFIX_LEN, _CACHE_LEN), dtype=torch.bool, device=device),
                    self.attention_out, _SCALE)

    def _slots(self, index: int):
        """The kernel's `(query, key_slot, value_slot)` views for one layer."""
        if self.fused_attention:
            return (self.query, self.key_cache[index][:, PREFIX_LEN:],
                    self.value_cache[index][:, PREFIX_LEN:])
        return (self.query.permute(1, 0, 2),
                self.key_cache[index, PREFIX_LEN:].permute(1, 0, 2),
                self.value_cache[index, PREFIX_LEN:].permute(1, 0, 2))

    def _invariants(self, prefix_masks):
        """Attention mask and RoPE table for this forward's valid prefix length."""
        from lingbotvla.models.vla.pi0.utils import make_att_2d_masks

        suffix = make_att_2d_masks(self.suffix_pad, self.suffix_att)
        prefix = prefix_masks[:, None, :].expand(prefix_masks.shape[0], SUFFIX_LEN, PREFIX_LEN)
        mask = torch.cat((prefix, suffix), dim=2)
        positions = (torch.sum(prefix_masks, dim=-1)[:, None]
                     + torch.cumsum(self.suffix_pad, dim=1) - 1)
        radians = torch.einsum(
            "bl,h->blh", positions.to(torch.float32), self.inverse_timescale)[..., None, :]
        return mask, torch.cos(radians), torch.sin(radians)

    def _embed(self, state_emb, noisy_actions, condition):
        """Upstream `embed_suffix` for the fixed one-state, fifty-action suffix."""
        action_emb = self.core.action_in_proj(noisy_actions)
        time_emb = condition.unsqueeze(1).expand(-1, action_emb.shape[1], -1)
        fused = self.core.action_time_mlp_in(torch.cat((action_emb, time_emb), dim=-1))
        fused = self.core.action_time_mlp_out(torch.nn.functional.silu(fused))
        return torch.cat((state_emb[:, None], fused), dim=1)

    def _project(self, layer, normed, index, cos, sin):
        """Project, rotate and cache one layer-step's q/k/v; returns the query view."""
        packed = torch.nn.functional.linear(
            normed, layer.packed_qkv_weight, layer.packed_qkv_bias)
        if self.fused_rope:
            self._kernels.rope_project(packed.view(SUFFIX_LEN, -1), cos[0, :, 0], sin[0, :, 0],
                                       *self._slots(index))
            return self.query
        packed = packed.float()
        query, key, value = packed.split(layer.qkv_widths, dim=-1)
        self.key_cache[index, PREFIX_LEN:] = _apply_rope(
            key.view(1, SUFFIX_LEN, KV_HEADS, HEAD_DIM), cos, sin)[0]
        self.value_cache[index, PREFIX_LEN:] = value.view(SUFFIX_LEN, KV_HEADS, HEAD_DIM)
        return _apply_rope(query.view(1, SUFFIX_LEN, -1, HEAD_DIM), cos, sin)[0]

    def _attention(self, query, index, mask):
        """One layer-step's masked GQA attention, as bf16 `[1, SUFFIX_LEN, H * D]`."""
        if self.attention_kernel:
            return self._kernels.fused_attention(
                query, self.key_cache[index], self.value_cache[index], mask[0],
                self.attention_out, _SCALE)[None]
        if not self.fused_attention:
            return _grouped_attention(query[None], self.key_cache[index][None],
                                      self.value_cache[index][None], mask)
        heads = query.shape[0]
        grouped = query.view(KV_HEADS, self.group * SUFFIX_LEN, HEAD_DIM)
        weights = torch.matmul(grouped, self.key_cache[index].transpose(-1, -2))
        self._kernels.masked_softmax(weights.view(heads, SUFFIX_LEN, _CACHE_LEN),
                                     mask[0], _SCALE)
        output = torch.matmul(weights, self.value_cache[index]).view(heads, SUFFIX_LEN, HEAD_DIM)
        self._kernels.attention_epilogue(output, self.attention_out)
        return self.attention_out[None]

    def _layer(self, layer, hidden, condition, index, mask, cos, sin):
        normed = layer.input_layernorm(hidden, condition)
        query = self._project(layer, normed, index, cos, sin)
        attention = self._attention(query, index, mask)

        weight_dtype = layer.self_attn.o_proj.weight.dtype
        out = layer.self_attn.o_proj(
            attention if attention.dtype == weight_dtype else attention.to(weight_dtype))
        out += hidden
        # The AdaRMS that follows returns a new tensor and never writes through
        # its input, so `out` stays available as the second residual.
        normed = layer.post_attention_layernorm(out, condition)
        gate_up = torch.nn.functional.linear(normed, layer.packed_gate_up_weight)
        gate_values, up_values = gate_up.split(layer.gate_up_widths, dim=-1)
        out += layer.mlp.down_proj(torch.nn.functional.silu(gate_values) * up_values)
        return out

    def _modulate(self, norm, source, residual, condition, total):
        """`total = source + residual`, normalized into `self.normed`, in one launch."""
        self._kernels.ada_rms_add(source.view(SUFFIX_LEN, -1), residual, norm.weight,
                                  norm.gamma(condition)[0], norm.beta(condition)[0],
                                  total, self.normed, norm.variance_epsilon)

    def _stack(self, hidden, condition, mask, cos, sin):
        """One denoise step through the whole stack, on the fused pointwise kernels.

        Each AdaRMS absorbs the residual add that precedes it and publishes the
        sum the next residual needs, so a layer issues twelve launches: four
        GEMMs, the five attention-block kernels, two modulations and the gated
        activation.
        """
        self._modulate(self.layers[0].input_layernorm, hidden, self.no_residual,
                       condition, self.hidden)
        for index, layer in enumerate(self.layers):
            query = self._project(layer, self.normed[None], index, cos, sin)
            attention = self._attention(query, index, mask)
            projected = layer.self_attn.o_proj(attention)
            self._modulate(layer.post_attention_layernorm, projected, self.hidden,
                           condition, self.residual)
            gate_up = torch.nn.functional.linear(self.normed, layer.packed_gate_up_weight)
            down = layer.mlp.down_proj(self._kernels.silu_multiply(gate_up, self.activation))
            if index + 1 == self.depth:
                return (down.view(SUFFIX_LEN, -1) + self.residual)[None]
            self._modulate(self.layers[index + 1].input_layernorm, down, self.residual,
                           condition, self.hidden)
        raise AssertionError("the expert stack always returns from its last layer")

    def _prime(self, prefix_k, prefix_v):
        """Copy this forward's prefix cache into the resident buffers, once."""
        if self.fused_attention:
            self.key_cache[:, :, :PREFIX_LEN] = prefix_k[:self.depth].permute(0, 2, 1, 3)
            self.value_cache[:, :, :PREFIX_LEN] = prefix_v[:self.depth].permute(0, 2, 1, 3)
            return
        self.key_cache[:, :PREFIX_LEN] = prefix_k[:self.depth]
        self.value_cache[:, :PREFIX_LEN] = prefix_v[:self.depth]

    @torch.no_grad()
    def run(self, state, noise, prefix_masks, prefix_k, prefix_v, actions, velocity_step_0):
        """Write the denoised chunk into `actions` and the first velocity into `velocity_step_0`.

        `prefix_k`/`prefix_v` are `[LAYERS, PREFIX_LEN, KV_HEADS, HEAD_DIM]` float32
        as the graph declares them; only the active depth is read.
        """
        self._prime(prefix_k, prefix_v)
        mask, cos, sin = self._invariants(prefix_masks)
        state_emb = self.core.state_proj(state)

        current = noise.clone()
        for step in range(self.steps):
            self.time_step[0] = step
            condition = self.conditions[step]
            hidden = self._embed(state_emb, current, condition)
            if self.fused_mlp:
                hidden = self._stack(hidden, condition, mask, cos, sin)
            else:
                for index, layer in enumerate(self.layers):
                    hidden = self._layer(layer, hidden, condition, index, mask, cos, sin)
            velocity = self.core.action_out_proj(self.final_norm(hidden)[:, -CHUNK:])
            if step == 0:
                velocity_step_0.copy_(velocity)
            current += self.dt * velocity
        actions.copy_(current)


__all__ = ["ExpertLoop"]
