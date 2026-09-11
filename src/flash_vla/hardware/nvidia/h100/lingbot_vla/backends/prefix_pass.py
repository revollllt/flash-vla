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

    def __init__(self, core, *, layers: int, scratch, device,
                 fused_pointwise: bool = False, attention_kernel: bool = False) -> None:
        self.core = core
        self.fused_pointwise = fused_pointwise
        self.attention_kernel = attention_kernel
        self.module = core.qwenvl_with_expert
        backbone = self.module.qwenvl.model
        self.layers = tuple(backbone.layers[:layers])
        self.final_norm = backbone.norm
        self.depth = layers

        attention = self.layers[0].self_attn
        self.query_heads = attention.q_proj.out_features // HEAD_DIM
        self.group = self.query_heads // KV_HEADS
        width = attention.q_proj.in_features
        for layer in self.layers:
            self._pack(layer, fused_pointwise)
        if fused_pointwise:
            for role in ("hidden", "residual", "normed"):
                setattr(self, role, scratch(f"lingbot_prefix_{role}", (PREFIX_LEN, width),
                                            torch.bfloat16, device))
            self.activation = scratch(
                "lingbot_prefix_activation",
                (PREFIX_LEN, self.layers[0].mlp.gate_proj.out_features),
                torch.bfloat16, device)
            # The first normalization of the pass has no residual yet; a zero
            # one keeps it on the same kernel, since bf16(x + 0) is exactly x.
            self.no_residual = scratch("lingbot_prefix_zero", (PREFIX_LEN, width),
                                       torch.bfloat16, device)
            self.no_residual.zero_()

        self.query = scratch("lingbot_prefix_query", (self.query_heads, PREFIX_LEN, HEAD_DIM),
                             torch.float32, device)
        self.attention_out = scratch("lingbot_prefix_attention",
                                     (PREFIX_LEN, self.query_heads * HEAD_DIM),
                                     torch.bfloat16, device)
        half = HEAD_DIM // 2
        exponents = (2.0 / HEAD_DIM) * torch.arange(half, dtype=torch.float32, device=device)
        self.inverse_timescale = 1.0 / (_ROPE_WAVELENGTH ** exponents)

        from .cuda import pointwise

        self._kernels = pointwise
        self._bind(device, half)

    @staticmethod
    def _pack(layer, gated: bool) -> None:
        """One packed q/k/v weight per layer, and optionally one packed gate/up.

        The backbone feed-forward is 11008 wide, so its packed copy costs 90 MB
        a layer; it is opt-in for that reason, not because the arithmetic
        differs -- concatenating along the output dimension leaves every dot
        product and its K reduction where they were.
        """
        attention = layer.self_attn
        layer.packed_qkv_weight = torch.cat(
            [attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight], dim=0)
        layer.packed_qkv_bias = torch.cat(
            [attention.q_proj.bias, attention.k_proj.bias, attention.v_proj.bias], dim=0)
        layer.qkv_width = sum(p.out_features for p in
                              (attention.q_proj, attention.k_proj, attention.v_proj))
        if gated:
            layer.packed_gate_up_weight = torch.cat(
                [layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight], dim=0)

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
        if self.attention_kernel:
            self._kernels.fused_attention(
                self.query,
                torch.zeros((KV_HEADS, PREFIX_LEN, HEAD_DIM), dtype=torch.float32, device=device),
                torch.zeros((KV_HEADS, PREFIX_LEN, HEAD_DIM), dtype=torch.float32, device=device),
                torch.ones((PREFIX_LEN, PREFIX_LEN), dtype=torch.bool, device=device),
                self.attention_out, _SCALE)
        if self.fused_pointwise:
            norm = self.layers[0].input_layernorm
            self._kernels.rms_norm_add(self.hidden, self.no_residual, norm.weight,
                                       self.residual, self.normed, norm.variance_epsilon)
            self._kernels.silu_multiply(
                torch.zeros((PREFIX_LEN, 2 * self.activation.shape[1]),
                            dtype=torch.bfloat16, device=device), self.activation)

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

    def _attend(self, layer, key, value, mask):
        """One layer's masked attention, into the shared bf16 output buffer."""
        if self.attention_kernel:
            self._kernels.fused_attention(self.query, key, value, mask[0],
                                          self.attention_out, _SCALE)
            return
        grouped = self.query.view(KV_HEADS, self.group * PREFIX_LEN, HEAD_DIM)
        weights = torch.matmul(grouped, key.transpose(-1, -2))
        self._kernels.masked_softmax(
            weights.view(self.query_heads, PREFIX_LEN, PREFIX_LEN), mask[0], _SCALE)
        output = torch.matmul(weights, value).view(self.query_heads, PREFIX_LEN, HEAD_DIM)
        self._kernels.attention_epilogue(output, self.attention_out)

    def _layer(self, layer, hidden, key, value, mask, cos, sin):
        self._project(layer, hidden, key, value, cos, sin)
        self._attend(layer, key, value, mask)
        out = layer.self_attn.o_proj(self.attention_out[None])
        out += hidden
        # `post_attention_layernorm` returns a new tensor and never writes
        # through its input, so `out` stays available as the second residual.
        out += layer.mlp(layer.post_attention_layernorm(out))
        return out

    def _normalize(self, norm, source, residual, total):
        """`total = source + residual`, normalized into `self.normed`, in one launch."""
        self._kernels.rms_norm_add(source.view(PREFIX_LEN, -1), residual, norm.weight,
                                   total, self.normed, norm.variance_epsilon)

    def _fused_stack(self, hidden, key_of, value_of, mask, cos, sin):
        """The whole stack on the fused pointwise kernels.

        Each normalization absorbs the residual add that precedes it and
        publishes the sum the next residual needs, so a layer issues nine
        launches: five GEMMs, the three attention-block kernels and the gated
        activation, plus one normalization.
        """
        self._normalize(self.layers[0].input_layernorm, hidden, self.no_residual, self.hidden)
        for index, layer in enumerate(self.layers):
            key, value = key_of(index), value_of(index)
            self._kernels.rope_project(
                torch.nn.functional.linear(self.normed, layer.packed_qkv_weight,
                                           layer.packed_qkv_bias),
                cos[0, :, 0], sin[0, :, 0], self.query, key, value)
            # This stage's outputs are the caches alone, so the last layer's
            # attention tail and feed-forward feed nothing.
            if index + 1 == self.depth:
                return
            self._attend(layer, key, value, mask)
            projected = layer.self_attn.o_proj(self.attention_out)
            self._normalize(layer.post_attention_layernorm, projected, self.hidden,
                            self.residual)
            gate_up = torch.nn.functional.linear(self.normed, layer.packed_gate_up_weight)
            down = layer.mlp.down_proj(
                self._kernels.silu_multiply(gate_up, self.activation))
            self._normalize(self.layers[index + 1].input_layernorm, down, self.residual,
                            self.hidden)

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
        if self.fused_pointwise:
            self._fused_stack(hidden, lambda i: prefix_k[i].permute(1, 0, 2),
                              lambda i: prefix_v[i].permute(1, 0, 2), mask, cos, sin)
            return
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
