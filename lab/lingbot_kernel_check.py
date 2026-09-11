"""Standalone check of the expert's hand-written stages against their torch paths.

    python -m lab.lingbot_rope_check

Runs on the Target's real shapes with random inputs; no checkpoint is needed.
"""
import torch

from flash_vla.hardware.nvidia.h100.lingbot_vla.backends.cuda import pointwise
from flash_vla.models.lingbot.spec import (
    HEAD_DIM, KV_HEADS, PREFIX_LEN, QUERY_HEADS, SUFFIX_LEN,
)

CACHE_LEN = PREFIX_LEN + SUFFIX_LEN
SCALE = HEAD_DIM ** -0.5


def _rope_reference(packed, cos, sin):
    half = HEAD_DIM // 2
    widths = [QUERY_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM]
    query, key, value = packed.float().split(widths, dim=-1)

    def rope(x, heads):
        x = x.view(SUFFIX_LEN, heads, HEAD_DIM)
        first, second = x.split(half, dim=-1)
        c, s = cos[:, None, :], sin[:, None, :]
        return torch.cat((first * c - second * s, second * c + first * s), dim=-1)

    return rope(query, QUERY_HEADS), rope(key, KV_HEADS), value.view(SUFFIX_LEN, KV_HEADS, HEAD_DIM)


def _report(name, got, want, tolerance=0.0):
    equal = torch.equal(got, want)
    error = (got.float() - want.float()).abs().max().item()
    print(f"  {name:22s} bitwise={str(equal):5s} max_abs={error:.3e}")
    return equal or error <= tolerance


def check_rope(device) -> bool:
    width = (QUERY_HEADS + 2 * KV_HEADS) * HEAD_DIM
    packed = torch.randn(SUFFIX_LEN, width, dtype=torch.bfloat16, device=device)
    angles = torch.randn(SUFFIX_LEN, HEAD_DIM // 2, device=device) * 10
    cos, sin = torch.cos(angles), torch.sin(angles)
    want_q, want_k, want_v = _rope_reference(packed, cos, sin)

    ok = True
    for label, head_major in (("token-major", False), ("head-major", True)):
        key_cache = torch.zeros(KV_HEADS, CACHE_LEN, HEAD_DIM, device=device)
        value_cache = torch.zeros_like(key_cache)
        if head_major:
            query = torch.zeros(QUERY_HEADS, SUFFIX_LEN, HEAD_DIM, device=device)
            slots = (query, key_cache[:, PREFIX_LEN:], value_cache[:, PREFIX_LEN:])
        else:
            token_cache = torch.zeros(CACHE_LEN, KV_HEADS, HEAD_DIM, device=device)
            value_token = torch.zeros_like(token_cache)
            query = torch.zeros(SUFFIX_LEN, QUERY_HEADS, HEAD_DIM, device=device)
            slots = (query.permute(1, 0, 2),
                     token_cache[PREFIX_LEN:].permute(1, 0, 2),
                     value_token[PREFIX_LEN:].permute(1, 0, 2))
        pointwise.rope_project(packed, cos, sin, *slots)
        torch.cuda.synchronize()
        if head_major:
            ok &= _report(f"rope/{label} query", query, want_q.permute(1, 0, 2))
            ok &= _report(f"rope/{label} key", key_cache[:, PREFIX_LEN:], want_k.permute(1, 0, 2))
            ok &= _report(f"rope/{label} value", value_cache[:, PREFIX_LEN:],
                          want_v.permute(1, 0, 2))
        else:
            ok &= _report(f"rope/{label} query", query, want_q)
            ok &= _report(f"rope/{label} key", token_cache[PREFIX_LEN:], want_k)
            ok &= _report(f"rope/{label} value", value_token[PREFIX_LEN:], want_v)
    return ok


def check_softmax(device) -> bool:
    scores = torch.randn(QUERY_HEADS, SUFFIX_LEN, CACHE_LEN, device=device) * 4
    mask = torch.rand(SUFFIX_LEN, CACHE_LEN, device=device) > 0.25
    mask[:, 0] = True
    want = torch.where(mask[None], scores * SCALE, -2.3819763e38)
    want = torch.nn.functional.softmax(want, dim=-1)
    got = scores.clone()
    pointwise.masked_softmax(got, mask, SCALE)
    torch.cuda.synchronize()
    return _report("masked softmax", got, want, tolerance=2e-7)


def check_epilogue(device) -> bool:
    source = torch.randn(QUERY_HEADS, SUFFIX_LEN, HEAD_DIM, device=device)
    target = torch.zeros(SUFFIX_LEN, QUERY_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device)
    pointwise.attention_epilogue(source, target)
    torch.cuda.synchronize()
    want = source.permute(1, 0, 2).reshape(SUFFIX_LEN, -1).to(torch.bfloat16)
    return _report("attention epilogue", target, want)


def check_rms_norm(device) -> bool:
    rows, width = 768, 1280
    source = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    weight = torch.randn(width, dtype=torch.bfloat16, device=device)
    target = torch.empty_like(source)
    pointwise.rms_norm(source, weight, target, 1e-6)
    torch.cuda.synchronize()
    values = source.float()
    values = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + 1e-6)
    want = weight * values.to(torch.bfloat16)
    return _report("rms norm", target, want, tolerance=8e-3)


def check_ada_rms_add(device) -> bool:
    rows, width = SUFFIX_LEN, 768
    source = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    residual = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    weight = torch.randn(width, dtype=torch.bfloat16, device=device)
    gamma = torch.randn(width, dtype=torch.bfloat16, device=device) * 0.1
    beta = torch.randn(width, dtype=torch.bfloat16, device=device) * 0.1
    total = torch.empty_like(source)
    target = torch.empty_like(source)
    pointwise.ada_rms_add(source, residual, weight, gamma, beta, total, target, 1e-6)
    torch.cuda.synchronize()

    want_total = source + residual
    values = want_total.float()
    values = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + 1e-6)
    values = weight * values
    want = ((1 + gamma.float()) * values + beta.float()).to(torch.bfloat16)
    return (_report("ada rms add total", total, want_total)
            & _report("ada rms add norm", target, want, tolerance=8e-3))


def check_silu_multiply(device) -> bool:
    rows, width = SUFFIX_LEN, 2752
    source = torch.randn(rows, 2 * width, dtype=torch.bfloat16, device=device)
    target = torch.empty(rows, width, dtype=torch.bfloat16, device=device)
    pointwise.silu_multiply(source, target)
    torch.cuda.synchronize()
    gate, up = source.split(width, dim=-1)
    want = torch.nn.functional.silu(gate) * up
    return _report("silu multiply", target, want, tolerance=8e-3)


def check_fused_attention(device) -> bool:
    heads, kv_heads, rows, keys, dim = QUERY_HEADS, KV_HEADS, SUFFIX_LEN, CACHE_LEN, HEAD_DIM
    group = heads // kv_heads
    query = torch.randn(heads, rows, dim, device=device)
    key = torch.randn(kv_heads, keys, dim, device=device)
    value = torch.randn(kv_heads, keys, dim, device=device)
    mask = torch.rand(rows, keys, device=device) > 0.25
    mask[:, 0] = True
    target = torch.zeros(rows, heads * dim, dtype=torch.bfloat16, device=device)
    pointwise.fused_attention(query, key, value, mask, target, SCALE)
    # The backbone hands the kernel a transposed view of the graph's own cache,
    # so the strided path is the one that actually ships.
    strided_key = key.permute(1, 0, 2).contiguous().permute(1, 0, 2)
    strided_value = value.permute(1, 0, 2).contiguous().permute(1, 0, 2)
    strided = torch.zeros_like(target)
    pointwise.fused_attention(query, strided_key, strided_value, mask, strided, SCALE)
    torch.cuda.synchronize()

    weights = torch.matmul(query.view(kv_heads, group * rows, dim),
                           key.transpose(-1, -2)).view(heads, rows, keys)
    weights = torch.where(mask[None], weights * SCALE, -2.3819763e38)
    probs = torch.nn.functional.softmax(weights, dim=-1)
    out = torch.matmul(probs.view(kv_heads, group * rows, keys), value)
    want = out.view(heads, rows, dim).permute(1, 0, 2).reshape(rows, -1).to(torch.bfloat16)
    return (_report("fused attention", target, want, tolerance=3.2e-2)
            & _report("fused attention strided", strided, want, tolerance=3.2e-2))


def main() -> int:
    torch.manual_seed(0)
    device = "cuda"
    ok = (check_rope(device) & check_softmax(device) & check_epilogue(device)
          & check_rms_norm(device) & check_ada_rms_add(device) & check_silu_multiply(device)
          & check_fused_attention(device))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
