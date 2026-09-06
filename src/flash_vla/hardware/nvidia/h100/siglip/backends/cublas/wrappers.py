"""cuBLASLt vision wrappers: LayerNorm, then a GEMM with a fused library epilogue.

The two pre-norm projections are the only vision sites whose epilogue cuBLASLt
fuses natively (bias; bias + tanh-GELU). Pi0.5 already reaches them through its
own TileLang backend's wrappers; Pi0 runs TileLang GEMMs at the same shapes and
pays 22.29 / 28.10 us per layer against Pi0.5's 16.50 / 19.11 (job 598964,
in-graph attribution, ACD1-55), which is 0.48 ms of the 0.49 ms gap between the
two Targets' vision stages. Sharing the form from this package lets Pi0 bank
that without waiting on a hand-written kernel, and leaves it superseded on both
Targets the moment the sm90 kernel beats cuBLASLt.

Every wrapper matches its `runtime.ops` signature positionally, writes only
`out`, allocates no device memory that outlives the call, and issues no driver
call, so all of them are safe inside CUDA-graph capture.
"""
from __future__ import annotations

import torch

from ... import geometry


def _layer_norm(x2, norm_w, norm_b):
    """LayerNorm over the feature axis of (M, DIM) bf16 -> (M, DIM) bf16.

    ATen accumulates the statistics in fp32 for a bf16 input and rounds once on
    output, which is the rounding point both Targets' shipped TileLang bodies
    use as well.

    The result is a fresh tensor rather than a `scratch` workspace, which is a
    deliberate exception to the rule that a backend materializing the
    normalized rows takes them from the injected allocator. torch exposes no
    allocation-free single-launch LayerNorm: `F.layer_norm` has no `out=`, and
    `aten.native_layer_norm.out` additionally wants mean and rstd buffers whose
    dtype differs between CPU and CUDA, so honouring the rule costs either a
    second 1.77 MB elementwise copy per call (~0.13 ms over the stage, a third
    of what this candidate is worth) or a device-dependent workspace key that
    would fail after the allocator freezes. Allocating here is what the shipped
    `vision_encoder_attention` and `vision_encoder_patch_embed` wrappers already
    do: capture places the tensor in the graph's private pool at a fixed
    address, so the replay path itself still allocates nothing. The sm90 kernel
    retires the question by fusing the norm into the GEMM prologue and
    materializing nothing at all.
    """
    return torch.nn.functional.layer_norm(
        x2, (geometry.DIM,), norm_w, norm_b, eps=geometry.NORM_EPS)


def vision_encoder_norm_qkv(x, norm_w, norm_b, qkv_w, qkv_b, out):
    """LayerNorm then the packed QKV projection.

    x (VIEWS, TOKENS, DIM) bf16 in; qkv_w (DIM, QKV_DIM), qkv_b (QKV_DIM,) bf16
    weights; out (VIEWS, TOKENS, QKV_DIM) bf16 written in place. Two launches.
    Graph-capture safe.
    """
    rows = x.shape[0] * geometry.TOKENS
    x_norm2 = _layer_norm(x.view(rows, geometry.DIM), norm_w, norm_b)
    # cuBLASLt with its bias epilogue; `out=` writes the graph buffer directly.
    # fp32 accumulate, bf16 out.
    torch.addmm(qkv_b, x_norm2, qkv_w, out=out.view(rows, qkv_w.shape[1]))
    return out


def vision_encoder_norm_ffn_up(x, norm_w, norm_b, weight, bias, out):
    """LayerNorm then the GELU feed-forward expansion.

    x (VIEWS, TOKENS, DIM) bf16 in; weight (DIM, FFN), bias (FFN,) bf16 weights;
    out (VIEWS, TOKENS, FFN) bf16 written in place. Two launches. Graph-capture
    safe.
    """
    rows = x.shape[0] * geometry.TOKENS
    x_norm2 = _layer_norm(x.view(rows, geometry.DIM), norm_w, norm_b)
    # CUBLASLT_EPILOGUE_GELU is the tanh approximation, i.e. upstream's
    # gelu_pytorch_tanh -- the erf form would be a different function, not a
    # rounding difference.
    torch._addmm_activation(bias, x_norm2, weight, use_gelu=True,
                            out=out.view(rows, geometry.FFN))
    return out


ALL_WRAPPERS = {
    "vision_encoder_norm_qkv": vision_encoder_norm_qkv,
    "vision_encoder_norm_ffn_up": vision_encoder_norm_ffn_up,
}

#: The call sites this backend implements.
NAMES = frozenset(ALL_WRAPPERS)

#: Each call site here is standalone: no route constraint, no extension op, and
#: so no graph contract and no `eval.smoke` route oracle to maintain.
ROUTE_CONSTRAINTS: tuple = ()
OPS: tuple = ()


def make_wrappers(scratch, selected_names=None) -> dict:
    """The wrappers of `selected_names` (default: all).

    `scratch` is unused: neither wrapper holds workspace across calls (see
    `_layer_norm`). It stays in the signature because it is the registry's
    contract, and the sm90 backend beside this one does use it.
    """
    del scratch
    names = NAMES if selected_names is None else set(selected_names)
    unknown = names - NAMES
    if unknown:
        raise KeyError(f"siglip cublas backend does not implement {sorted(unknown)}")
    return {name: ALL_WRAPPERS[name] for name in names}
