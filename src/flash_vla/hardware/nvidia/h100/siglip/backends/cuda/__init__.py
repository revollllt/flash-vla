"""Hand-written sm90 CUDA vision backend.

Exposes the backend contract the runtime's registry consumes. `wrappers` stays
importable on its own so a parity script can call the functions directly;
`siglip_attn` and `siglip_norm` own their builds and launches; and
`attention_reference` and `norm_gemm_reference` are the T2 ABI mirrors the
wrappers are checked against.
"""
from __future__ import annotations

from . import attention_reference, siglip_attn, siglip_norm, wrappers
from ..cublas import norm_gemm_reference
from .wrappers import NAMES, OPS, ROUTE_CONSTRAINTS, make_wrappers

__all__ = ["NAMES", "OPS", "ROUTE_CONSTRAINTS", "attention_reference",
           "make_wrappers", "norm_gemm_reference", "siglip_attn", "siglip_norm",
           "wrappers"]
