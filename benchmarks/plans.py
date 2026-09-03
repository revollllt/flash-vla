"""The named call-site plans, and how a plan name is resolved.

A plan is the provenance record of a run: it maps each call site the pipeline
touches to the backend that implemented it, and `Pi05Inference` overlays nothing
on top of it. So a report that carries the plan says exactly what ran, and two
numbers are comparable when their plans are.

`tilelang` is the reference route -- every call site on TileLang, encoder
attention on the torch chain -- and is what `plan_parity` compares a candidate
against. Every other plan here runs the promoted CUDA encoder attention, so a
latency comparison against `tilelang` moves the prefix as well as the decoder;
the `-enc-tilelang` variant exists to A/B that one call site with the decoder
held fixed.

This module holds no device dependency on purpose: the offline trace analysis in
`layer_breakdown` reads a plan to decide which kernel sequence to expect, and
must not have to import torch to do it.
"""
from __future__ import annotations

import json

#: The promoted encoder attention: one fused CUDA kernel in place of the
#: QK^T / softmax / PV torch chain
#: (`.agents/notes/implemented/feature/2026-09-03-encoder-mqa-cuda-attention.md`).
_ENC_ATTN_CUDA = {"encoder_attention": "cuda"}

#: Named plans for `--plan`. A JSON object is accepted too.
PLANS = {
    "tilelang": None,
    "attn-cuda": {
        **_ENC_ATTN_CUDA,
        "decoder_norm_qkv_rope": "cuda",
        "decoder_attention": "cuda",
    },
    "ffn-cuda": {
        **_ENC_ATTN_CUDA,
        "decoder_norm_gated_ffn": "cuda",
        "decoder_ffn_down_residual": "cuda",
    },
    "ffn-cuda-fused-producer": {
        **_ENC_ATTN_CUDA,
        "decoder_out_proj_residual": "cuda",
        "decoder_norm_gated_ffn": "cuda",
        "decoder_ffn_down_residual": "cuda",
    },
    "attn-ffn-cuda": {
        **_ENC_ATTN_CUDA,
        "decoder_norm_qkv_rope": "cuda",
        "decoder_attention": "cuda",
        "decoder_norm_gated_ffn": "cuda",
        "decoder_ffn_down_residual": "cuda",
    },
    "attn-ffn-cuda-fused-producer": {
        **_ENC_ATTN_CUDA,
        "decoder_norm_qkv_rope": "cuda",
        "decoder_attention": "cuda",
        "decoder_out_proj_residual": "cuda",
        "decoder_norm_gated_ffn": "cuda",
        "decoder_ffn_down_residual": "cuda",
    },
    # The A/B leg for the encoder attention: the same decoder route with that one
    # call site back on the reference chain. Both legs run in one process
    # (`--plan a --plan b --plan a`), which is the A/B/A design the promotion
    # rests on -- node and clock drift cancel only within a job.
    "attn-ffn-cuda-fused-producer-enc-tilelang": {
        "decoder_norm_qkv_rope": "cuda",
        "decoder_attention": "cuda",
        "decoder_out_proj_residual": "cuda",
        "decoder_norm_gated_ffn": "cuda",
        "decoder_ffn_down_residual": "cuda",
    },
    # PDL-chain variants: same routes through the cuda-pdl backend. -pdlffn
    # arms only the FFN half (role-split wait releases the weight loaders
    # under the XFS producer); -pdl additionally chains rms -> qkv ->
    # attention -> combine with early triggers and waits at the first
    # dependent read. The encoder kernel is one launch between two TileLang
    # neighbours and arms no programmatic dependency, so it stays on "cuda".
    "attn-ffn-cuda-fused-producer-pdlffn": {
        **_ENC_ATTN_CUDA,
        "decoder_norm_qkv_rope": "cuda",
        "decoder_attention": "cuda",
        "decoder_out_proj_residual": "cuda-pdl",
        "decoder_norm_gated_ffn": "cuda-pdl",
        "decoder_ffn_down_residual": "cuda-pdl",
    },
    "attn-ffn-cuda-fused-producer-pdl": {
        **_ENC_ATTN_CUDA,
        "decoder_norm_qkv_rope": "cuda-pdl",
        "decoder_attention": "cuda-pdl",
        "decoder_out_proj_residual": "cuda-pdl",
        "decoder_norm_gated_ffn": "cuda-pdl",
        "decoder_ffn_down_residual": "cuda-pdl",
    },
}


def parse_plan(text: str) -> dict[str, str] | None:
    """A plan name from `PLANS`, or a JSON object mapping call site to backend."""
    if text in PLANS:
        return PLANS[text]
    return json.loads(text)


def encoder_attention_route(plan: dict[str, str] | None) -> str:
    """The backend a plan runs the encoder attention on.

    A plan that does not name the call site gets the table's default backend,
    TileLang, which is the torch chain.
    """
    return (plan or {}).get("encoder_attention", "tilelang")
