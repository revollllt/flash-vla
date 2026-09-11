"""The backbone's single prefix pass, written for this Target's fixed shapes.

The prefix fill runs the shared `QwenvlWithExpertModel` forward once over 264
tokens and 36 layers. That path is written for the general case, so per layer it
widens q, k and v to float32 separately, concatenates three single-element
lists, rebuilds the RoPE table, materializes eight `repeat`ed copies of the key
and of the value so every query head owns one, and finally hands the result
back through a cache dictionary that the call site copies into the graph's
output buffers.

None of that is needed here: the key and value buffers the graph declares are
exactly what the layer produces, and the shapes never change. This module runs
the same arithmetic writing straight into those buffers, and shares the action
expert's hand-written projection and softmax kernels.
"""
from __future__ import annotations

import torch

from flash_vla.models.lingbot.spec import (
    HEAD_DIM,
    KV_HEADS,
    PREFIX_LEN,
)

_ROPE_WAVELENGTH = 10_000.0
_SCALE = HEAD_DIM ** -0.5


class PrefixPass:
    """One capture-safe prefix fill over the backbone stack.

    Built during runner warmup, so its packed weights and workspace are
    allocated before the static arena freezes. `run` writes only into the
    graph's `prefix_masks`, `prefix_k` and `prefix_v` buffers and into buffers
    bound here.
    """

    def __init__(self, core, *, layers: int, scratch, device) -> None:
        self.core = core
        self.module = core.qwenvl_with_expert
        backbone = self.module.qwenvl.model
        self.layers = tuple(backbone.layers[:layers])
        self.final_norm = backbone.norm
        self.depth = layers

        attention = self.layers[0].self_attn
        self.query_heads = attention.q_proj.out_features // HEAD_DIM
        self.group = self.query_heads // KV_HEADS
        for layer in self.layers:
            self._pack(layer)

        self.query = scratch("lingbot_prefix_query", (self.query_heads, PREFIX_LEN, HEAD_DIM),
                             torch.float32, device)
        self.attention_out = scratch("lingbot_prefix_attention",
                                     (PREFIX_LEN, self.query_heads * HEAD_DIM),
                                     torch.bfloat16, device)
        half = HEAD_DIM // 2
        exponents = (2.0 / HEAD_DIM) * torch.arange(half, dtype=torch.float32, device=device)
        self.inverse_timescale = 1.0 / (_ROPE_WAVELENGTH ** exponents)

        from .cuda import expert_rope

        self._kernels = expert_rope
        self._bind(device, half)

    @staticmethod
    def _pack(layer) -> None:
        """One packed q/k/v weight per layer; the dot products are unchanged."""
        attention = layer.self_attn
        layer.packed_qkv_weight = torch.cat(
            [attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight], dim=0)
        layer.packed_qkv_bias = torch.cat(
            [attention.q_proj.bias, attention.k_proj.bias, attention.v_proj.bias], dim=0)
        layer.qkv_width = sum(p.out_features for p in
                              (attention.q_proj, attention.k_proj, attention.v_proj))

    def _bind(self, device, half: int) -> None:
        """Load the CUDA library and exercise it once, before capture begins."""
        self._kernels.rope_project(
            torch.zeros((PREFIX_LEN, self.layers[0].qkv_width),
                        dtype=torch.bfloat16, device=device),
            torch.zeros((PREFIX_LEN, half), dtype=torch.float32, device=device),
            torch.zeros((PREFIX_LEN, half), dtype=torch.float32, device=device),
            self.query,
            torch.zeros((KV_HEADS, PREFIX_LEN, HEAD_DIM), dtype=torch.float32, device=device),
            torch.zeros((KV_HEADS, PREFIX_LEN, HEAD_DIM), dtype=torch.float32, device=device))
        self._kernels.masked_softmax(
            torch.zeros((1, PREFIX_LEN, PREFIX_LEN), dtype=torch.float32, device=device),
            torch.ones((PREFIX_LEN, PREFIX_LEN), dtype=torch.bool, device=device), _SCALE)
        self._kernels.attention_epilogue(self.query, self.attention_out)

    def _embed(self, vision, image_masks, language_tokens, language_masks, prefix_masks):
        """Prefix embeddings and the padding mask the whole model shares."""
        from lingbotvla.models.vla.pi0.utils import make_att_2d_masks

        image_prefix = vision.reshape(1, -1, vision.shape[-1])
        tokens_per_view = image_prefix.shape[1] // image_masks.shape[0]
        expanded = image_masks.unsqueeze(0).repeat_interleave(tokens_per_view, dim=1)
        language = self.module.embed_language_tokens(language_tokens)
        embeddings = torch.cat((image_prefix, language), dim=1)
        masks = torch.cat((expanded, language_masks), dim=1)
        prefix_masks.copy_(masks)
        attention = make_att_2d_masks(masks, torch.zeros_like(masks))
        positions = torch.cumsum(masks, dim=1) - 1
        radians = torch.einsum(
            "bl,h->blh", positions.to(torch.float32), self.inverse_timescale)[..., None, :]
        return embeddings, attention, torch.cos(radians), torch.sin(radians)

    def _project(self, layer, hidden, key, value, cos, sin) -> None:
        """Write one layer's rotated key and its value into the graph's caches."""
        normed = layer.input_layernorm(hidden)
        packed = torch.nn.functional.linear(
            normed, layer.packed_qkv_weight, layer.packed_qkv_bias)
        self._kernels.rope_project(packed.view(PREFIX_LEN, -1), cos[0, :, 0], sin[0, :, 0],
                                   self.query, key, value)

    def _layer(self, layer, hidden, key, value, mask, cos, sin):
        self._project(layer, hidden, key, value, cos, sin)
        grouped = self.query.view(KV_HEADS, self.group * PREFIX_LEN, HEAD_DIM)
        weights = torch.matmul(grouped, key.transpose(-1, -2))
        self._kernels.masked_softmax(
            weights.view(self.query_heads, PREFIX_LEN, PREFIX_LEN), mask[0], _SCALE)
        output = torch.matmul(weights, value).view(self.query_heads, PREFIX_LEN, HEAD_DIM)
        self._kernels.attention_epilogue(output, self.attention_out)

        out = layer.self_attn.o_proj(self.attention_out[None])
        out += hidden
        # `post_attention_layernorm` returns a new tensor and never writes
        # through its input, so `out` stays available as the second residual.
        out += layer.mlp(layer.post_attention_layernorm(out))
        return out

    @torch.no_grad()
    def run(self, vision, image_masks, language_tokens, language_masks,
            prefix_masks, prefix_k, prefix_v):
        """Fill `prefix_masks`, `prefix_k` and `prefix_v` for one observation.

        `prefix_k`/`prefix_v` are `[LAYERS, PREFIX_LEN, KV_HEADS, HEAD_DIM]`
        float32 as the graph declares them; the rotated keys and the values are
        written straight into them, head-major through a transposed view.
        """
        hidden, mask, cos, sin = self._embed(
            vision, image_masks, language_tokens, language_masks, prefix_masks)
        for index, layer in enumerate(self.layers):
            key = prefix_k[index].permute(1, 0, 2)
            value = prefix_v[index].permute(1, 0, 2)
            # The last layer's attention tail and feed-forward feed only the
            # hidden state, and this stage's outputs are the caches alone.
            if index == self.depth - 1:
                self._project(layer, hidden, key, value, cos, sin)
                return
            hidden = self._layer(layer, hidden, key, value, mask, cos, sin)


__all__ = ["PrefixPass"]
