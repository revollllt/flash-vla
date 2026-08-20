"""TileLang tuning adapter: the parts of a sweep that are not portable.

The sweep loop itself is in `flash_vla.tuning`. What stays here is everything
that only means something to TileLang -- `tilelang.jit`, `pass_configs`, the raw
builder registry -- plus the translation from a device capability to a tuning
axis, which is the piece people expect to be a hardware property and is not.

Warp specialization is the case that shows why. It is a TileLang lowering
strategy: `T.copy` becomes TMA plus a producer/consumer warp split. So the axis
exists only where TMA does, and `axes()` reads `SUPPORTS_TMA` to decide. But a
hand-written CUDA backend on the same H100 would use TMA directly with no such
flag, and on a device without TMA would reach for `cp.async` instead -- same
hardware fact, entirely different axis. That mapping belongs to a backend, which
is why `flash_vla.tuning` never sees it and `H100Spec` never hears the phrase
"warp specialization".

Whether the axis is worth sweeping is a separate question from whether it is
legal, and both matter: below one wave the producer warp has no work to hide and
costs warps plus mbarrier traffic, so small decoder GEMMs want it off while the
encoder and vision stages at M=768 want it on. It is also not universally legal
even on H100 -- the fused dual-GEMM gate reuses one shared tile across two
weight stages, which the no-WS pipeline planner rejects at compile time. Those
candidates are counted and skipped, not fatal.

Kernels are re-wrapped from `kernels.RAW_KERNELS` rather than reconfigured in
place: `.compile(pass_configs=...)` is rejected as unhashable, and a decorated
kernel does not expose its underlying builder.
"""
from __future__ import annotations

import tilelang

from flash_vla.hardware.nvidia.h100.spec import H100Spec
from flash_vla.tuning import grid, sweep

from .kernels import base as kernels


def rewrap(name: str, warp_spec: bool):
    """Re-wrap the raw builder for `name` with warp specialization on or off."""
    raw, out_idx = kernels.RAW_KERNELS[name]
    pass_configs = kernels.FAST_MATH if warp_spec else kernels.NO_WARP_SPEC
    if out_idx == "default":
        return tilelang.jit(raw, pass_configs=pass_configs)
    return tilelang.jit(raw, out_idx=out_idx, pass_configs=pass_configs)


def axes(spec=H100Spec, **tile_axes) -> list[dict]:
    """Candidate set for `spec`: the caller's tile axes crossed with warp specialization.

    Without TMA the axis collapses to a single False rather than disappearing,
    so it still shows up in every result row and in the report -- a config table
    should say the flag was considered and forced, not leave it unmentioned.

    `spec` is a parameter rather than a hardcoded import so the derivation can be
    read and tested against a different device without one existing yet. It is
    not a portability claim: a second device gets its own target directory, and
    the two adapters only merge if they turn out identical.
    """
    warp_spec = (True, False) if spec.SUPPORTS_TMA else (False,)
    return grid(warp_spec=warp_spec, **tile_axes)


def builder(name: str, const_kwargs: dict):
    """Return the `build` callable `tuning.sweep` needs for kernel `name`.

    Splits `warp_spec` back out of the candidate: it selects the wrapper, while
    everything else is a compile-time constant of the kernel itself.
    """
    def build(candidate: dict):
        config = dict(candidate)
        warp_spec = config.pop("warp_spec")
        return rewrap(name, warp_spec).compile(**const_kwargs, **config)

    return build


def sweep_kernel(name: str, const_kwargs: dict, tile_axes: dict, invoke, *, spec=H100Spec, **kwargs):
    """Sweep one kernel over its tile axes and the warp-specialization flag.

    const_kwargs   shape constants passed to every compile (M, N, K, HEAD_DIM, ...)
    tile_axes      tunable axes as lists: BLOCK_M=[16, 32], NUM_STAGES=[2, 3], ...
    invoke(k, i)   issue one launch of compiled kernel k against the i-th cold input set

    Remaining keyword arguments go to `tuning.sweep` -- `correct=` in particular,
    which a re-tune must not skip: `kernels.tl_scaled_gate` is numerically
    sensitive to its tiling and some tilings produce garbage rather than failing.
    """
    return sweep(axes(spec, **tile_axes), builder(name, const_kwargs), invoke,
                 label=name, **kwargs)


# ---------------------------------------------------------------------------
# Call-site sweeps
#
# `sweep_kernel` tunes one kernel. Some decisions do not live in one kernel:
# FlashDecoding's split count speeds up the split and slows down the combine at
# the same time, because more splits means more partials to merge. Ranking the
# split kernel alone would pick the largest split count every time and make the
# call site slower. So this sweeps the *pair*, through the same generic loop --
# `tuning.sweep` only asks that `build` return something `invoke` can call.
# ---------------------------------------------------------------------------


def sweep_decoder_attention(m_flat: int, head_dim: int, keys: int, mask,
                            split_requests=(6, 8, 12, 16, 24, 32),
                            block_n=(32, 64), device: str = "cuda", **kwargs):
    """Sweep FlashDecoding's requested split count and key-block width together.

    `NUM_SPLIT` is a *request*: `wrappers._num_splits` shrinks it until no split
    starts at or past `keys`, because a TMA box whose first row is out of bounds
    reads garbage rather than zeros. So several requests collapse onto the same
    realized split count; they are deduplicated here rather than measured twice.

    The correctness check is not optional. It is also the only thing that would
    catch a split count whose partials merge wrongly, which is silent -- the
    combine weights an empty split by an lse of about -9e36 and produces a
    plausible number rather than a NaN.
    """
    import torch

    from flash_vla.tuning import cold_n_inner, grid, sweep

    from . import wrappers as w
    from .kernels import adarms as ada_kernels
    from .kernels import base as kernels

    generator = torch.Generator(device=device).manual_seed(0)

    def rand(*shape):
        return (torch.randn(shape, generator=generator, device=device,
                            dtype=torch.float32) * 0.05).bfloat16()

    footprint = 2 * keys * head_dim * 2          # K and V, what the next call will not reuse
    n_inner = min(cold_n_inner(footprint, H100Spec.L2_CACHE_SIZE_BYTES), 64)
    inputs = [(rand(m_flat, head_dim), rand(keys, head_dim), rand(keys, head_dim))
              for _ in range(n_inner)]
    outputs = [torch.empty_like(inputs[0][0]) for _ in range(n_inner)]

    q0, k0, v0 = inputs[0]
    logits = (q0.float() @ k0.float().T) * (head_dim ** -0.5) + mask.float()[None, :]
    expected = (torch.softmax(logits, dim=-1) @ v0.float())

    seen: set[tuple[int, int]] = set()

    def feasible(candidate: dict) -> bool:
        realized = w._num_splits(keys, candidate["BLOCK_N"], candidate["NUM_SPLIT"])
        key = (candidate["BLOCK_N"], realized[0])
        if key in seen:
            return False
        seen.add(key)
        return True

    def build(candidate: dict):
        block_n_value, requested = candidate["BLOCK_N"], candidate["NUM_SPLIT"]
        num_split, chunk_blocks = w._num_splits(keys, block_n_value, requested)
        q_pad = -(-m_flat // w._FD_SPLIT["BLOCK_M"]) * w._FD_SPLIT["BLOCK_M"]
        partial = torch.empty((num_split, q_pad, head_dim), dtype=torch.bfloat16, device=device)
        glse = torch.empty((num_split, q_pad), dtype=torch.float32, device=device)

        split_fn = ada_kernels.tl_fd_flat_split_mask.compile(
            M=m_flat, HD=head_dim, KEYS=keys, QPAD=q_pad, NUM_SPLIT=num_split,
            BLOCK_M=w._FD_SPLIT["BLOCK_M"], BLOCK_N=block_n_value,
            NUM_STAGES=w._FD_SPLIT["NUM_STAGES"], THREADS=w._FD_SPLIT["THREADS"],
            CHUNK=chunk_blocks * block_n_value, CHUNK_BLOCKS=chunk_blocks,
            SCALE_L2=float(head_dim ** -0.5) * w._LOG2E)
        combine_fn = kernels.tl_fd_flat_combine.compile(
            M=m_flat, HD=head_dim, QPAD=q_pad, NUM_SPLIT=num_split,
            BLOCK_M=w._FD_COMBINE_BLOCK_M, THREADS=128)

        def launch(index: int):
            q, k, v = inputs[index % n_inner]
            split_fn(q, k, v, mask, partial, glse)
            combine_fn(partial, glse, outputs[index % n_inner])

        launch.realized_split = num_split
        launch.ctas = -(-m_flat // w._FD_SPLIT["BLOCK_M"]) * num_split
        return launch

    def correct(launch) -> bool:
        launch(0)
        torch.cuda.synchronize()
        got = outputs[0].float()
        return bool(torch.nn.functional.cosine_similarity(
            expected.flatten(), got.flatten(), dim=0).item() > 0.999)

    candidates = grid(BLOCK_N=list(block_n), NUM_SPLIT=list(split_requests))
    return sweep(candidates, build, lambda launch, i: launch(i), feasible=feasible,
                 correct=correct, label="decoder_attention", n_inner=n_inner, **kwargs)


def sweep_decoder_qkv(m: int, n: int, k: int, head_dim: int, num_heads: int,
                      block_n=(8, 16, 32), block_k=(64, 128, 256), num_stages=(2, 3, 4),
                      device: str = "cuda", **kwargs):
    """Sweep the AdaRMS QKV projection, whose grid is `N / BLOCK_N` at M=50.

    One m-tile of 64 covers the whole action chunk, so BLOCK_N alone decides how
    much of the machine this kernel occupies: 80 CTAs at 32, 160 at 16, 320 at 8.
    """
    import torch

    from flash_vla.tuning import cold_n_inner

    generator = torch.Generator(device=device).manual_seed(0)

    def rand(*shape):
        return (torch.randn(shape, generator=generator, device=device,
                            dtype=torch.float32) * 0.05).bfloat16()

    n_inner = min(cold_n_inner(k * n * 2, H100Spec.L2_CACHE_SIZE_BYTES), 32)
    weights = [rand(k, n) for _ in range(n_inner)]
    x, factor, scale = rand(m, k), rand(m), rand(k)
    bias, rope = rand(n), rand(m, head_dim)
    out_q = torch.empty((m, num_heads * head_dim), dtype=torch.bfloat16, device=device)
    out_k = torch.empty((m, head_dim), dtype=torch.bfloat16, device=device)
    out_v = torch.empty((m, head_dim), dtype=torch.bfloat16, device=device)

    scaled = (x * scale[None, :]).float() @ weights[0].float()
    expected = scaled * factor.float()[:, None] + bias.float()[None, :]

    def correct(compiled) -> bool:
        compiled(x, factor, scale, weights[0], bias, rope, out_q, out_k, out_v)
        torch.cuda.synchronize()
        got = torch.cat([out_q, out_k, out_v], dim=1).float()
        # RoPE rotates Q and K, so only the V columns compare directly; a wrong
        # tiling shows up there just as loudly and needs no rotation reference.
        return bool(torch.isfinite(got).all().item() and
                    torch.nn.functional.cosine_similarity(
                        expected[:, -head_dim:].flatten(),
                        got[:, -head_dim:].flatten(), dim=0).item() > 0.999)

    def invoke(compiled, index):
        compiled(x, factor, scale, weights[index % n_inner], bias, rope, out_q, out_k, out_v)

    return sweep_kernel("tl_ada_qkv_gemm_rope",
                        dict(M=m, N=n, K=k, HEAD_DIM=head_dim, NUM_HEADS=num_heads),
                        dict(BLOCK_M=[64], BLOCK_N=list(block_n), BLOCK_K=list(block_k),
                             NUM_STAGES=list(num_stages), THREADS=[128]),
                        invoke, correct=correct, n_inner=n_inner, **kwargs)


def main(argv=None) -> int:
    """Re-tune the decoder attention call site at this target's real shapes.

    Run as a module, not through `benchmarks`: a benchmark may only consume the
    public engine API, and this reaches into the backend on purpose.

        python -m flash_vla.hardware.nvidia.h100.pi05.backends.tilelang.autotune
    """
    import argparse

    import torch

    from flash_vla.models.pi05.spec import DECODER_HEADS, HEAD_DIM, MAX_TOKEN_LEN, VISION_TOKENS

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--prompt-tokens", type=int, default=135,
                        help="valid prompt tokens; only moves where the mask hole sits")
    parser.add_argument("--site", default="attention", choices=("attention", "qkv"))
    args = parser.parse_args(argv)

    if args.site == "qkv":
        from flash_vla.models.pi05.spec import DECODER_DIM, QKV_WIDTH
        print(f"# decoder qkv: M={args.chunk_size} N={QKV_WIDTH} K={DECODER_DIM}")
        outcome = sweep_decoder_qkv(args.chunk_size, QKV_WIDTH, DECODER_DIM,
                                    HEAD_DIM, DECODER_HEADS)
        if outcome.best:
            print(f"\n# best: {outcome.best}")
        return 0

    image_tokens = args.num_views * VISION_TOKENS
    keys = image_tokens + MAX_TOKEN_LEN + args.chunk_size
    n_valid = image_tokens + args.prompt_tokens

    mask = torch.zeros((keys,), dtype=torch.bfloat16, device="cuda")
    mask[n_valid:image_tokens + MAX_TOKEN_LEN] = -3.0e38

    print(f"# decoder attention: M_flat={args.chunk_size * DECODER_HEADS} HD={HEAD_DIM} "
          f"keys={keys}, mask hole [{n_valid}, {image_tokens + MAX_TOKEN_LEN})")
    outcome = sweep_decoder_attention(args.chunk_size * DECODER_HEADS, HEAD_DIM, keys, mask)
    if outcome.best:
        print(f"\n# best: BLOCK_N={outcome.best['BLOCK_N']} "
              f"NUM_SPLIT request {outcome.best['NUM_SPLIT']} -> {outcome.best['us']:.2f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
