"""The framework's op vocabulary: one `OpSpec` per call site.

A VLA forward pass is written as an explicit graph over a fixed vocabulary of
call sites (`runtime/graph.py`). Each call site has one spec: the positional
parameter names every backend's wrapper for it must take in this order, which
of those parameters the wrapper writes in place, which are weights, which are
auxiliary outputs that no cost model counts, and the FLOP formula over the
argument shapes. The spec is what lets the runner allocate, route, execute,
attribute and cost a graph without knowing the model.

Three stages, full names, shared by every Target:

    vision_encoder_*   patch embedding and the SigLIP transformer
    llm_backbone_*     projection into the language model, prompt embedding,
                       the prefix transformer that builds the KV cache
    action_expert_*    the flow-matching action expert over the KV cache

A Target's backend registry may add extension ops (a state projection only
one model has) with the same structure. Parameters a model does not use are
passed as `None`; a backend wrapper asserts what it needs.

Every op writes at least one output parameter in place and its return value
is ignored by the runner. Scalars (an int prefix length, a float scale) are
plain Python values in the argument list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any, Callable, Mapping

from .cost import Cost

Shapes = Mapping[str, Any]


def _rows(shape) -> int:
    return prod(shape[:-1])


def gemm(x: str, w: str, *, out: str | None = None) -> Callable[[Shapes], int]:
    """2 * M * K * N with M from `x` (or `out`) rows, K from the weight's leading dims, N its last."""
    def flops(s: Shapes) -> int:
        m = _rows(s[out] if out is not None else s[x])
        return 2 * m * prod(s[w][:-1]) * s[w][-1]
    return flops


def dual_gemm(x: str, w: str) -> Callable[[Shapes], int]:
    """Two M x K @ K x N GEMMs sharing the activation (a gated FFN's gate and up)."""
    def flops(s: Shapes) -> int:
        return 2 * (2 * _rows(s[x]) * prod(s[w][:-1]) * s[w][-1])
    return flops


def attention(q: str, k: str) -> Callable[[Shapes], int]:
    """QK^T and PV: 4 * query_rows * keys * head_dim, query rows counting every head."""
    def flops(s: Shapes) -> int:
        return 4 * s[q][0] * s[k][0] * s[q][-1]
    return flops


def packed_attention(qkv: str) -> Callable[[Shapes], int]:
    """Bidirectional attention within each view of a packed (views, tokens, 3 * dim) buffer."""
    def flops(s: Shapes) -> int:
        views, tokens, width = s[qkv]
        return 4 * views * tokens * tokens * (width // 3)
    return flops


@dataclass(frozen=True)
class OpSpec:
    """The contract of one call site.

    `params` is the wrapper's positional parameter list. `outputs` are the
    parameters the wrapper writes; `inout` the subset it also reads (a residual
    accumulated in place); `weights` the model weights among the parameters;
    `aux` the outputs that exist for the implementation's convenience (a
    materialized normalized activation, a per-row norm factor) and count
    neither as traffic nor as a stage contract. `flops` maps parameter shapes
    to floating-point operations; `bytes_override` replaces the default
    traffic rule for ops whose reads are not their arguments' sizes (a gather).
    """
    name: str
    params: tuple[str, ...]
    outputs: tuple[str, ...]
    weights: tuple[str, ...] = ()
    inout: tuple[str, ...] = ()
    aux: tuple[str, ...] = ()
    flops: Callable[[Shapes], int] | None = None
    bytes_override: Callable[[Shapes, Mapping[str, int]], tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        names = set(self.params)
        for group in (self.outputs, self.weights, self.inout, self.aux):
            unknown = set(group) - names
            if unknown:
                raise ValueError(f"{self.name}: {sorted(unknown)} not in params {self.params}")
        if not self.outputs:
            raise ValueError(f"{self.name}: an op must write at least one output")

    def cost(self, shapes: Shapes, itemsizes: Mapping[str, int]) -> Cost:
        """Minimal traffic and math of one invocation with these argument shapes.

        Reads are every tensor argument that is not an output, plus the
        in-place residuals; writes are the outputs; auxiliary outputs are
        neither. Missing (`None`) arguments cost nothing.
        """
        def nbytes(name: str) -> int:
            shape = shapes.get(name)
            return prod(shape) * itemsizes[name] if shape is not None else 0

        if self.bytes_override is not None:
            read, written = self.bytes_override(shapes, itemsizes)
        else:
            outputs = set(self.outputs)
            aux = set(self.aux)
            read = sum(nbytes(p) for p in self.params if p not in outputs and p not in aux)
            read += sum(nbytes(p) for p in self.inout)
            written = sum(nbytes(p) for p in self.outputs if p not in aux)
        flops = self.flops(shapes) if self.flops is not None else 0
        return Cost(bytes_read=read, bytes_written=written, flops=flops)


def _gather_bytes(shapes: Shapes, itemsizes: Mapping[str, int]) -> tuple[int, int]:
    """A row gather reads as many table rows as it writes."""
    written = prod(shapes["out"]) * itemsizes["out"]
    return written, written


#: The standard vocabulary. Parameter order is the wrapper signature.
STANDARD: tuple[OpSpec, ...] = (
    # --- vision encoder ---------------------------------------------------
    OpSpec("vision_encoder_patch_embed",
           ("images", "patch_w", "patch_b", "pos_emb", "out"), outputs=("out",),
           weights=("patch_w", "patch_b", "pos_emb"), flops=gemm("images", "patch_w", out="out")),
    OpSpec("vision_encoder_norm_qkv",
           ("x", "norm_w", "norm_b", "qkv_w", "qkv_b", "out"),
           outputs=("out",), weights=("norm_w", "norm_b", "qkv_w", "qkv_b"),
           flops=gemm("x", "qkv_w")),
    OpSpec("vision_encoder_attention", ("qkv", "out"), outputs=("out",),
           flops=packed_attention("qkv")),
    OpSpec("vision_encoder_out_proj_residual",
           ("x", "weight", "bias", "res", "out"), outputs=("out",),
           weights=("weight", "bias"), flops=gemm("x", "weight")),
    OpSpec("vision_encoder_norm_ffn_up",
           ("x", "norm_w", "norm_b", "weight", "bias", "out"),
           outputs=("out",), weights=("norm_w", "norm_b", "weight", "bias"),
           flops=gemm("x", "weight")),
    OpSpec("vision_encoder_ffn_down_residual",
           ("x", "weight", "bias", "res", "out"), outputs=("out",),
           weights=("weight", "bias"), flops=gemm("x", "weight")),
    # --- llm backbone -----------------------------------------------------
    OpSpec("llm_backbone_projector",
           ("x", "norm_w", "norm_b", "proj_w", "proj_b", "out", "x_norm"),
           outputs=("out", "x_norm"), weights=("norm_w", "norm_b", "proj_w", "proj_b"),
           aux=("x_norm",), flops=gemm("x", "proj_w")),
    OpSpec("llm_backbone_embed_prompt",
           ("token_ids", "table", "scale", "out"), outputs=("out",), weights=("table",),
           bytes_override=_gather_bytes),
    OpSpec("llm_backbone_norm_qkv_rope",
           ("x", "weight_qkv", "rope", "q", "k", "v", "x_norm"),
           outputs=("q", "k", "v", "x_norm"), weights=("weight_qkv",), aux=("x_norm",),
           flops=gemm("x", "weight_qkv")),
    OpSpec("llm_backbone_attention",
           ("q", "k", "v", "scale", "mask", "out"), outputs=("out",),
           flops=attention("q", "k")),
    OpSpec("llm_backbone_out_proj_residual",
           ("x", "weight", "out"), outputs=("out",), inout=("out",), weights=("weight",),
           flops=gemm("x", "weight")),
    OpSpec("llm_backbone_norm_gated_ffn",
           ("x", "gate_w", "up_w", "out", "x_norm"), outputs=("out", "x_norm"),
           weights=("gate_w", "up_w"), aux=("x_norm",), flops=dual_gemm("x", "gate_w")),
    OpSpec("llm_backbone_ffn_down_residual",
           ("x", "weight", "out"), outputs=("out",), inout=("out",), weights=("weight",),
           flops=gemm("x", "weight")),
    # --- action expert ----------------------------------------------------
    OpSpec("action_expert_action_in_proj",
           ("x", "weight", "bias", "out"), outputs=("out",), weights=("weight", "bias"),
           flops=gemm("x", "weight")),
    OpSpec("action_expert_norm_qkv_rope",
           ("x", "scale", "weight_qkv", "bias", "rope", "q", "k", "v", "norm_factor"),
           outputs=("q", "k", "v", "norm_factor"), weights=("scale", "weight_qkv", "bias"),
           aux=("norm_factor",), flops=gemm("x", "weight_qkv")),
    OpSpec("action_expert_attention",
           ("q", "k", "v", "mask", "out", "prefix_len"), outputs=("out",),
           flops=attention("q", "k")),
    OpSpec("action_expert_out_proj_residual",
           ("x", "weight", "gate", "out"), outputs=("out",), inout=("out",),
           weights=("weight", "gate"), flops=gemm("x", "weight")),
    OpSpec("action_expert_norm_gated_ffn",
           ("x", "scale", "gate_w", "up_w", "gate_b", "up_b", "out", "norm_factor"),
           outputs=("out", "norm_factor"), weights=("scale", "gate_w", "up_w", "gate_b", "up_b"),
           aux=("norm_factor",), flops=dual_gemm("x", "gate_w")),
    OpSpec("action_expert_ffn_down_residual",
           ("x", "weight", "gate", "out"), outputs=("out",), inout=("out",),
           weights=("weight", "gate"), flops=gemm("x", "weight")),
    OpSpec("action_expert_action_out_proj",
           ("x", "weight", "bias", "out", "norm_factor"), outputs=("out", "norm_factor"),
           inout=("out",), weights=("weight", "bias"), aux=("norm_factor",),
           flops=gemm("x", "weight")),
)


class Vocabulary:
    """The specs a graph is checked against: the standard set plus a Target's extensions."""

    def __init__(self, extra: tuple[OpSpec, ...] = ()) -> None:
        self._specs: dict[str, OpSpec] = {spec.name: spec for spec in STANDARD}
        for spec in extra:
            if spec.name in self._specs:
                raise ValueError(f"extension op {spec.name!r} shadows a standard op")
            self._specs[spec.name] = spec

    def __getitem__(self, name: str) -> OpSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(f"unknown call site {name!r}; known: {sorted(self._specs)}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)


__all__ = ["OpSpec", "STANDARD", "Vocabulary", "attention", "dual_gemm", "gemm",
           "packed_attention"]
