"""The framework's op vocabulary: one `OpSpec` per call site.

A VLA forward pass is written as an explicit graph over a fixed vocabulary of
call sites (`runtime/graph.py`). Each call site has one spec: the positional
parameter names every backend's wrapper for it must take in this order, which
of those parameters the wrapper writes in place and which are weights. The
spec is what lets the runner allocate, route, execute and attribute a graph
without knowing the model.

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

from dataclasses import dataclass

@dataclass(frozen=True)
class OpSpec:
    """The contract of one call site.

    `params` is the wrapper's positional parameter list. `outputs` are the
    parameters the wrapper writes; `inout` the subset it also reads (a residual
    accumulated in place); `weights` the model weights among the parameters.
    What a call site costs is not declared here: `measurement.work` reads it
    from the model's reference.
    """
    name: str
    params: tuple[str, ...]
    outputs: tuple[str, ...]
    weights: tuple[str, ...] = ()
    inout: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = set(self.params)
        for group in (self.outputs, self.weights, self.inout):
            unknown = set(group) - names
            if unknown:
                raise ValueError(f"{self.name}: {sorted(unknown)} not in params {self.params}")
        if not self.outputs:
            raise ValueError(f"{self.name}: an op must write at least one output")


#: The standard vocabulary. Parameter order is the wrapper signature.
STANDARD: tuple[OpSpec, ...] = (
    # --- vision encoder ---------------------------------------------------
    OpSpec("vision_encoder_patch_embed",
           ("images", "patch_w", "patch_b", "pos_emb", "out"), outputs=("out",),
           weights=("patch_w", "patch_b", "pos_emb")),
    OpSpec("vision_encoder_norm_qkv",
           ("x", "norm_w", "norm_b", "qkv_w", "qkv_b", "out"),
           outputs=("out",), weights=("norm_w", "norm_b", "qkv_w", "qkv_b")),
    OpSpec("vision_encoder_attention", ("qkv", "out"), outputs=("out",)),
    OpSpec("vision_encoder_out_proj_residual",
           ("x", "weight", "bias", "res", "out"), outputs=("out",),
           weights=("weight", "bias")),
    OpSpec("vision_encoder_norm_ffn_up",
           ("x", "norm_w", "norm_b", "weight", "bias", "out"),
           outputs=("out",), weights=("norm_w", "norm_b", "weight", "bias")),
    OpSpec("vision_encoder_ffn_down_residual",
           ("x", "weight", "bias", "res", "out"), outputs=("out",),
           weights=("weight", "bias")),
    # --- llm backbone -----------------------------------------------------
    OpSpec("llm_backbone_projector",
           ("x", "norm_w", "norm_b", "proj_w", "proj_b", "out", "x_norm"),
           outputs=("out", "x_norm"), weights=("norm_w", "norm_b", "proj_w", "proj_b")),
    OpSpec("llm_backbone_embed_prompt",
           ("token_ids", "table", "scale", "out"), outputs=("out",), weights=("table",)),
    OpSpec("llm_backbone_norm_qkv_rope",
           ("x", "weight_qkv", "rope", "q", "k", "v", "x_norm"),
           outputs=("q", "k", "v", "x_norm"), weights=("weight_qkv",)),
    OpSpec("llm_backbone_attention",
           ("q", "k", "v", "scale", "mask", "out"), outputs=("out",)),
    OpSpec("llm_backbone_out_proj_residual",
           ("x", "weight", "out"), outputs=("out",), inout=("out",), weights=("weight",)),
    OpSpec("llm_backbone_norm_gated_ffn",
           ("x", "gate_w", "up_w", "out", "x_norm"), outputs=("out", "x_norm"),
           weights=("gate_w", "up_w")),
    OpSpec("llm_backbone_ffn_down_residual",
           ("x", "weight", "out"), outputs=("out",), inout=("out",), weights=("weight",)),
    # --- action expert ----------------------------------------------------
    OpSpec("action_expert_action_in_proj",
           ("x", "weight", "bias", "out"), outputs=("out",), weights=("weight", "bias")),
    OpSpec("action_expert_norm_qkv_rope",
           ("x", "scale", "weight_qkv", "bias", "rope", "q", "k", "v", "norm_factor"),
           outputs=("q", "k", "v", "norm_factor"), weights=("scale", "weight_qkv", "bias")),
    OpSpec("action_expert_attention",
           ("q", "k", "v", "mask", "out", "prefix_len"), outputs=("out",)),
    OpSpec("action_expert_out_proj_residual",
           ("x", "weight", "gate", "out"), outputs=("out",), inout=("out",),
           weights=("weight", "gate")),
    OpSpec("action_expert_norm_gated_ffn",
           ("x", "scale", "gate_w", "up_w", "gate_b", "up_b", "out", "norm_factor"),
           outputs=("out", "norm_factor"), weights=("scale", "gate_w", "up_w", "gate_b", "up_b")),
    OpSpec("action_expert_ffn_down_residual",
           ("x", "weight", "gate", "out"), outputs=("out",), inout=("out",),
           weights=("weight", "gate")),
    OpSpec("action_expert_action_out_proj",
           ("x", "weight", "bias", "out", "norm_factor"), outputs=("out", "norm_factor"),
           inout=("out",), weights=("weight", "bias")),
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


__all__ = ["OpSpec", "STANDARD", "Vocabulary"]
