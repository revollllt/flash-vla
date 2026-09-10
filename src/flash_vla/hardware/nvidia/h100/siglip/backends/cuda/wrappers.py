"""Hand-written sm90 vision wrappers.

Three call sites. The fused attention kernel replaces the production route's
three launches (a cuDNN workspace memset, the SDPA kernel, and the transpose
and copy back into the graph buffer) with one. The two pre-norm projections
keep cuBLASLt's fused bias and tanh-GELU epilogues, which no hand-written GEMM
here beat, and replace only the LayerNorm ahead of them: torch's generic
`F.layer_norm` measured about 3.4 us/call slower than the tuned TileLang body
on this shape (job 599790), which is what made the pure-torch route a
regression on Pi0.5.

The four GEMM call sites are deliberately absent, and that is a measured
decision rather than an omission. A prior lane built a hand-written sm90 GEMM
on this same tile library at these same shapes; it beat cuBLAS at one site
only, by less than the promotion bar, and its ablation showed the copy column
rather than the math is the floor, with the per-transaction TMA issue cost
applying per SM. Folding the LayerNorm into the GEMM's A side needs
BLOCK_M <= 64, which raises that box count rather than lowering it. See the
Agent Note.

Every wrapper matches its `runtime.ops` signature positionally, writes only
`out`, allocates nothing on the replay path, and issues no driver call, so all
of them are safe inside CUDA-graph capture.
"""
from __future__ import annotations

from functools import partial

import torch

from ... import geometry
from . import siglip_attn, siglip_norm


def fresh_scratch(role: str, shape, dtype, device) -> torch.Tensor:
    """The default workspace allocator: a new zeroed tensor per call.

    Only for calling a wrapper as a free function outside a runner. A runner
    injects its own allocator through `make_wrappers`, which allocates each
    workspace once and freezes it before capture.
    """
    del role
    return torch.zeros(shape, dtype=dtype, device=device)


def _norm_workspace(scratch, x2):
    """The normalized rows the two pre-norm GEMMs consume.

    Workspace of this backend, not a graph buffer: the ops stopped declaring an
    `x_norm` output in PR0 so that a backend which materializes the normalized
    rows takes them from the injected allocator and one that fuses the norm
    writes nothing. Unlike torch's `F.layer_norm`, this kernel writes a
    caller-provided buffer, so the rule is honoured here.
    """
    return scratch("vision_norm", tuple(x2.shape), x2.dtype, x2.device)


def vision_encoder_attention(qkv, out):
    """Multi-head self-attention over the packed QKV buffer.

    qkv (VIEWS, TOKENS, 3*DIM) bf16 in; out (VIEWS, TOKENS, DIM) bf16 written
    in place. One launch. Graph-capture safe.
    """
    siglip_attn.attention(qkv, out)
    return out


def vision_encoder_norm_qkv(x, norm_w, norm_b, qkv_w, qkv_b, out, *, scratch=fresh_scratch):
    """LayerNorm then the packed QKV projection. Two launches.

    x (VIEWS, TOKENS, DIM) bf16 in; qkv_w (DIM, QKV_DIM), qkv_b (QKV_DIM,) bf16
    weights; out (VIEWS, TOKENS, QKV_DIM) bf16 written in place.
    """
    rows = x.shape[0] * geometry.TOKENS
    x2 = x.view(rows, geometry.DIM)
    x_norm2 = siglip_norm.layer_norm(x2, norm_w, norm_b, _norm_workspace(scratch, x2))
    # cuBLASLt with its bias epilogue; `out=` writes the graph buffer directly.
    torch.addmm(qkv_b, x_norm2, qkv_w, out=out.view(rows, qkv_w.shape[1]))
    return out


def vision_encoder_norm_ffn_up(x, norm_w, norm_b, weight, bias, out, *, scratch=fresh_scratch):
    """LayerNorm then the GELU feed-forward expansion. Two launches.

    x (VIEWS, TOKENS, DIM) bf16 in; weight (DIM, FFN), bias (FFN,) bf16
    weights; out (VIEWS, TOKENS, FFN) bf16 written in place.
    """
    rows = x.shape[0] * geometry.TOKENS
    x2 = x.view(rows, geometry.DIM)
    x_norm2 = siglip_norm.layer_norm(x2, norm_w, norm_b, _norm_workspace(scratch, x2))
    # CUBLASLT_EPILOGUE_GELU is the tanh approximation, i.e. upstream's
    # gelu_pytorch_tanh -- the erf form would be a different function.
    torch._addmm_activation(bias, x_norm2, weight, use_gelu=True,
                            out=out.view(rows, geometry.FFN))
    return out


ALL_WRAPPERS = {
    "vision_encoder_attention": vision_encoder_attention,
    "vision_encoder_norm_qkv": vision_encoder_norm_qkv,
    "vision_encoder_norm_ffn_up": vision_encoder_norm_ffn_up,
}

#: The call sites this backend implements.
NAMES = frozenset(ALL_WRAPPERS)

#: Wrappers that materialize the normalized rows and take the runner's allocator.
_NEEDS_SCRATCH = frozenset({"vision_encoder_norm_qkv", "vision_encoder_norm_ffn_up"})

#: Each call site is standalone: the attention kernel reads the packed QKV
#: buffer as the graph declares it, so it composes with any routing of the
#: projection that produces it. No route constraint, no extension op, no graph
#: contract, and no `tests.targets` route oracle to maintain.
ROUTE_CONSTRAINTS: tuple = ()
OPS: tuple = ()


def make_wrappers(scratch, selected_names=None) -> dict:
    """The wrappers of `selected_names` (default: all), bound to `scratch`."""
    names = NAMES if selected_names is None else set(selected_names)
    unknown = names - NAMES
    if unknown:
        raise KeyError(f"siglip cuda backend does not implement {sorted(unknown)}")
    return {name: (partial(ALL_WRAPPERS[name], scratch=scratch)
                   if name in _NEEDS_SCRATCH else ALL_WRAPPERS[name])
            for name in names}
