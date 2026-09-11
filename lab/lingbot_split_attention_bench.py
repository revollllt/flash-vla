"""Correctness and latency of the split-key attention kernel at the real shape.

    PYTHONPATH=src:. python -m lab.lingbot_split_attention_bench --reps 400
    ... --configurations 40:64:1:4:2:0,32:64:1:2:2:1 --prefix

A configuration is "keys:rows:groups:unroll:pieces:tensor"; the default sweep is
the shipped float32 shape followed by the TF32 ones. `--prefix` adds the
backbone's 264-row shape, which the kernel is correct at and slower than cuBLAS
at. Needs a GPU, so it runs under sbatch; no checkpoint, the inputs are random
at the Target's fixed shape.

Correctness is checked three ways, because the interesting failures are not the
ones a tolerance catches. `check` compares against the torch chain at both
matmul precisions and against a float64 reference, over a contiguous cache, a
strided cache view and a fully masked row. `check_sentinel` pins the finite
-2.3819763e38 sentinel path to exact answers -- a unit value cache, where every
row must come out exactly 1.0 under any mask, so a miscounted key is a bit
difference rather than 3.2e-3 of drift hiding inside a bf16 ulp. `phases` reads
the SM clock at each phase boundary, which is what attributed the kernel's cost
to staging rather than to arithmetic.

Latency is a median over CUDA-graph replays, so the numbers include the
per-launch cost the deployed graph pays. The chain is timed at both
allow_tf32 settings: upstream's eager attention issues TF32 tensor-core GEMMs,
so the TF32 row is the baseline that matters and torch's float32 default is not.
"""
from __future__ import annotations

import argparse
import statistics

import torch

from flash_vla.hardware.nvidia.h100.lingbot_vla.backends.cuda import pointwise, split_attention
from flash_vla.models.lingbot.spec import (
    HEAD_DIM, KV_HEADS, PREFIX_LEN, QUERY_HEADS, SUFFIX_LEN,
)

CACHE_LEN = PREFIX_LEN + SUFFIX_LEN
SCALE = HEAD_DIM ** -0.5
SENTINEL = -2.3819763e38

# The denominators the design is measured against. Bytes: the query, the two
# caches, the mask and the bf16 output, each touched once. MACs: QK plus PV.
BYTES = (QUERY_HEADS * SUFFIX_LEN * HEAD_DIM * 4 + 2 * KV_HEADS * CACHE_LEN * HEAD_DIM * 4
         + SUFFIX_LEN * CACHE_LEN + SUFFIX_LEN * QUERY_HEADS * HEAD_DIM * 2)
MACS = 2 * QUERY_HEADS * SUFFIX_LEN * CACHE_LEN * HEAD_DIM
STREAMING_FLOOR_US = 2.4
FMA_ROOFLINE_US = 2.2


def with_tf32(enabled):
    """Context for the cuBLAS precision the deployed route actually runs at.

    The upstream eager attention this is validated against issues TF32
    tensor-core GEMMs (`sm80_xmma_gemm_f32f32_tf32f32_f32` and
    `cutlass_80_tensorop_s1688gemm`, job 614222), while torch's own default for
    float32 matmul is the slower full-precision path -- so the chain has to be
    priced both ways or the comparison is against a baseline nobody deploys.
    """
    import contextlib

    @contextlib.contextmanager
    def scope():
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = enabled
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    return scope()


def reference(query, key, value, mask, dtype=torch.float32):
    """The torch chain the kernel replaces, as `[rows, heads * dim]` in `dtype`."""
    heads, rows, dim = query.shape
    kv_heads, keys, _ = key.shape
    group = heads // kv_heads
    q = query.to(dtype).view(kv_heads, group * rows, dim)
    weights = torch.matmul(q, key.to(dtype).transpose(-1, -2)).view(heads, rows, keys)
    weights = torch.where(mask[None], weights * SCALE, torch.tensor(SENTINEL, dtype=dtype,
                                                                   device=weights.device))
    probs = torch.nn.functional.softmax(weights, dim=-1)
    out = torch.matmul(probs.view(kv_heads, group * rows, keys), value.to(dtype))
    return out.view(heads, rows, dim).permute(1, 0, 2).reshape(rows, -1)


def _errors(got, want):
    a, b = got.float(), want.float()
    max_abs = (a - b).abs().max().item()
    rel_rms = ((a - b).pow(2).sum().sqrt() / b.pow(2).sum().sqrt()).item()
    cosine = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    return max_abs, rel_rms, cosine


def check(device, tile, rows_per_cta, groups, unroll, pieces, tensor) -> bool:
    torch.manual_seed(0)
    query = torch.randn(QUERY_HEADS, SUFFIX_LEN, HEAD_DIM, device=device)
    key = torch.randn(KV_HEADS, CACHE_LEN, HEAD_DIM, device=device)
    value = torch.randn(KV_HEADS, CACHE_LEN, HEAD_DIM, device=device)
    mask = torch.rand(SUFFIX_LEN, CACHE_LEN, device=device) > 0.25
    mask[:, 0] = True
    # Upstream's sentinel exists so that a row with nothing to attend to still
    # produces a uniform distribution; the split has to reproduce that across
    # slices, so one row is fully masked and one keeps a single live key.
    mask[3] = False
    mask[7] = False
    mask[7, CACHE_LEN - 1] = True

    # The backbone hands the kernel a transposed view of the graph's own cache,
    # so the strided path is the one that actually ships.
    strided_key = key.permute(1, 0, 2).contiguous().permute(1, 0, 2)
    strided_value = value.permute(1, 0, 2).contiguous().permute(1, 0, 2)

    exact = reference(query, key, value, mask, dtype=torch.float64)
    with with_tf32(False):
        chain = reference(query, key, value, mask).to(torch.bfloat16)
    with with_tf32(True):
        chain_tf32 = reference(query, key, value, mask).to(torch.bfloat16)
    uniform = value.mean(dim=1).repeat_interleave(QUERY_HEADS // KV_HEADS, dim=0)

    ok = True
    for label, k, v in (("contiguous", key, value), ("strided", strided_key, strided_value)):
        target = torch.zeros(SUFFIX_LEN, QUERY_HEADS * HEAD_DIM, dtype=torch.bfloat16,
                             device=device)
        split_attention.fused_attention(query, k, v, mask, target, SCALE, key_tile=tile,
                                        row_tile=rows_per_cta, row_groups=groups,
                                        unroll=unroll, pieces=pieces, tensor=tensor)
        torch.cuda.synchronize()
        against_chain = _errors(target, chain)
        against_tf32 = _errors(target, chain_tf32)
        against_exact = _errors(target, exact)
        drift = (target[3].float().view(QUERY_HEADS, HEAD_DIM) - uniform).abs().max().item()
        # One bf16 ulp of a unit-scale value; the chain itself rounds to the
        # same grid, so agreement to a couple of ulps is the whole claim
        # available.
        passed = against_chain[0] <= 3.2e-2 and against_chain[1] <= 2e-3 and drift <= 3.2e-2
        ok &= passed
        tag = ("tf32 " if tensor else "fp32 ") + f"keys={tile}"
        print(f"  {tag:16s} {label:10s} vs fp32 chain max_abs={against_chain[0]:.3e} "
              f"rel_rms={against_chain[1]:.3e} cos={against_chain[2]:.9f} "
              f"masked_row={drift:.3e}  {'PASS' if passed else 'FAIL'}")
        print(f"  {'':16s} {'':10s} vs tf32 chain max_abs={against_tf32[0]:.3e} "
              f"rel_rms={against_tf32[1]:.3e} cos={against_tf32[2]:.9f}")
        print(f"  {'':16s} {'':10s} vs float64   max_abs={against_exact[0]:.3e} "
              f"rel_rms={against_exact[1]:.3e} cos={against_exact[2]:.9f}")
    return ok


def check_sentinel(device, tile, rows_per_cta, groups, unroll, pieces, tensor) -> bool:
    """The finite-sentinel path, checked so that a wrong key count cannot pass.

    Upstream's masked logit is -2.3819763e38, not -inf, so a row with nothing to
    attend to must come out as the plain mean of the value cache rather than
    NaN. Across a split that holds only if every slice reports the sentinel as
    its maximum and its own live key count as its denominator, and if the keys
    past the last slice's 35 live ones contribute a hard zero rather than
    exp(0) = 1.

    A tolerance cannot establish that: one key miscounted out of 315 is 3.2e-3
    of relative error, inside two bf16 ulps. So the decisive case uses a value
    cache of all ones, where the answer is exactly 1.0 for every row under every
    mask -- the weights cancel against their own sum -- and any key counted into
    a denominator but not its accumulator (or the reverse) shows up as an exact
    bit difference rather than as drift.
    """
    torch.manual_seed(1)
    query = torch.randn(QUERY_HEADS, SUFFIX_LEN, HEAD_DIM, device=device)
    key = torch.randn(KV_HEADS, CACHE_LEN, HEAD_DIM, device=device)
    group = QUERY_HEADS // KV_HEADS
    # The Target's 315 keys are 7 full slices and a short one, so the final
    # slice's padded keys are the ones at risk.
    last_slice = (CACHE_LEN // tile) * tile

    live = torch.rand(SUFFIX_LEN, CACHE_LEN, device=device) > 0.25
    live[:, 0] = True
    live[3] = False                                  # nothing live anywhere
    live[7] = False
    live[7, CACHE_LEN - 1] = True                    # only the last key of the short slice
    live[11] = False
    live[11, 0] = True                               # only the first key of the first slice
    live[13] = False
    live[13, last_slice:] = True                     # only the short slice
    live[17] = False
    live[17, tile:2 * tile] = True                   # only one interior slice

    def run(value, mask):
        target = torch.full((SUFFIX_LEN, QUERY_HEADS * HEAD_DIM), float("nan"),
                            dtype=torch.bfloat16, device=device)
        split_attention.fused_attention(query, key, value, mask, target, SCALE, key_tile=tile,
                                        row_tile=rows_per_cta, row_groups=groups,
                                        unroll=unroll, pieces=pieces, tensor=tensor)
        torch.cuda.synchronize()
        return target

    ok = True
    ones = torch.ones(KV_HEADS, CACHE_LEN, HEAD_DIM, device=device)
    for label, mask in (("adversarial mask", live),
                        ("every row masked", torch.zeros_like(live))):
        target = run(ones, mask)
        want = torch.ones_like(target)
        exact = bool(torch.equal(target, want))
        wrong = int((target != want).sum())
        ok &= exact
        print(f"  unit cache, {label:17s} exact=1.0 everywhere: {exact} "
              f"(differing elements {wrong}, max_abs "
              f"{(target.float() - 1.0).abs().max().item():.3e})  "
              f"{'PASS' if exact else 'FAIL'}")

    value = torch.randn(KV_HEADS, CACHE_LEN, HEAD_DIM, device=device)
    target = run(value, live)
    finite = bool(torch.isfinite(target.float()).all())
    ok &= finite
    print(f"  random cache, all outputs finite: {finite}")

    # A fully masked row is the unweighted mean of the whole cache; one bf16 ulp
    # at that magnitude is ~2e-4, and the value is small enough that rounding
    # cannot disguise a miscount the unit-cache case would have caught anyway.
    uniform = value.double().mean(dim=1).repeat_interleave(group, dim=0)
    drift = (target[3].double().view(QUERY_HEADS, HEAD_DIM) - uniform).abs().max().item()
    passed = drift <= 1e-3
    ok &= passed
    print(f"  random cache, fully masked row = cache mean: max_abs={drift:.3e}  "
          f"{'PASS' if passed else 'FAIL'}")

    # A row with one live key is that key's value. The float32 path has no
    # arithmetic left to round differently from the torch chain and matches
    # bitwise; the TF32 path first rounds the value to a 10-bit mantissa, which
    # can carry it across a bf16 tie, so one ulp of the output's own scale is
    # the correct bound there.
    for row, index in ((7, CACHE_LEN - 1), (11, 0)):
        want = value[:, index].repeat_interleave(group, dim=0).to(torch.bfloat16)
        got = target[row].view(QUERY_HEADS, HEAD_DIM)
        difference = (got.float() - want.float()).abs()
        exact = bool(torch.equal(got, want))
        # One bf16 ulp, not one relative-precision step: bf16 keys eight
        # significand bits, so the spacing at a value in [2^e, 2^(e+1)) is
        # 2^e * 2^-7 -- twice the relative bound, and the difference between
        # calling a correct TF32 result correct and calling it a failure.
        allowed = float((want.float().abs() * 2.0 ** -7).clamp(min=2.0 ** -11).max())
        passed = exact or (tensor and float(difference.max()) <= allowed)
        ok &= passed
        print(f"  random cache, row {row:2d} live only at key {index:3d}: bitwise={exact}, "
              f"max_abs={float(difference.max()):.3e} (<= {allowed:.3e})  "
              f"{'PASS' if passed else 'FAIL'}")
    return ok


def prefix_shape(device, reps, inner, tile, rows_per_cta, groups, unroll, pieces, tensor):
    """The same operation at the backbone's prefix shape, for routing decisions."""
    rows = keys = PREFIX_LEN
    torch.manual_seed(2)
    query = torch.randn(QUERY_HEADS, rows, HEAD_DIM, device=device)
    key = torch.randn(KV_HEADS, keys, HEAD_DIM, device=device)
    value = torch.randn(KV_HEADS, keys, HEAD_DIM, device=device)
    mask = torch.rand(rows, keys, device=device) > 0.25
    mask[:, 0] = True
    mask[5] = False
    target = torch.zeros(rows, QUERY_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device)
    buffers = split_attention.workspace(QUERY_HEADS, rows, HEAD_DIM, keys, device, tile)

    split_attention.fused_attention(query, key, value, mask, target, SCALE, buffers=buffers,
                                    key_tile=tile, row_tile=rows_per_cta, row_groups=groups,
                                    unroll=unroll, pieces=pieces, tensor=tensor)
    torch.cuda.synchronize()
    chain_out = reference(query, key, value, mask).to(torch.bfloat16)
    errors = _errors(target, chain_out)
    uniform = value.double().mean(dim=1).repeat_interleave(QUERY_HEADS // KV_HEADS, dim=0)
    drift = (target[5].double().view(QUERY_HEADS, HEAD_DIM) - uniform).abs().max().item()
    finite = bool(torch.isfinite(target.float()).all())
    splits = split_attention.splits_for(keys, tile)
    tiles = -(-rows // rows_per_cta)
    print(f"  rows={rows} keys={keys}: {QUERY_HEADS * splits * tiles} CTAs, "
          f"workspace={buffers[0].numel() * 4 / 2**20:.1f} MiB")
    print(f"  vs chain max_abs={errors[0]:.3e} rel_rms={errors[1]:.3e} cos={errors[2]:.9f} "
          f"masked_row={drift:.3e} finite={finite}")

    grouped = query.view(KV_HEADS, (QUERY_HEADS // KV_HEADS) * rows, HEAD_DIM)
    key_t = key.transpose(-1, -2)
    weights = torch.empty(KV_HEADS, grouped.shape[1], keys, device=device)
    context = torch.empty(KV_HEADS, grouped.shape[1], HEAD_DIM, device=device)

    def chain_full():
        torch.bmm(grouped, key_t, out=weights)
        pointwise.masked_softmax(weights.view(QUERY_HEADS, rows, keys), mask, SCALE)
        torch.bmm(weights, value, out=context)
        pointwise.attention_epilogue(context.view(QUERY_HEADS, rows, HEAD_DIM), target)

    def run():
        split_attention.fused_attention(query, key, value, mask, target, SCALE,
                                        buffers=buffers, key_tile=tile, row_tile=rows_per_cta,
                                        row_groups=groups, unroll=unroll, pieces=pieces,
                                        tensor=tensor)

    with with_tf32(True):
        chain = _median_us(chain_full, reps, inner, device)
    ours = _median_us(run, reps, inner, device)
    print(f"  chain (4 launches) {chain:7.2f} us   split-key (2 launches) {ours:7.2f} us   "
          f"{chain / ours:.2f}x")
    return errors[0] <= 3.2e-2 and drift <= 8e-3 and finite


def _median_us(build, reps, inner, device):
    """Median microseconds per operation under CUDA-graph replay."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            build()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            build()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(reps):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / inner)
    return statistics.median(samples)


def benchmark(device, configurations, reps, inner):
    torch.manual_seed(0)
    group = QUERY_HEADS // KV_HEADS
    query = torch.randn(QUERY_HEADS, SUFFIX_LEN, HEAD_DIM, device=device)
    key = torch.randn(KV_HEADS, CACHE_LEN, HEAD_DIM, device=device)
    value = torch.randn(KV_HEADS, CACHE_LEN, HEAD_DIM, device=device)
    mask = torch.rand(SUFFIX_LEN, CACHE_LEN, device=device) > 0.25
    mask[:, 0] = True
    target = torch.zeros(SUFFIX_LEN, QUERY_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device)

    grouped = query.view(KV_HEADS, group * SUFFIX_LEN, HEAD_DIM)
    key_t = key.transpose(-1, -2)
    weights = torch.empty(KV_HEADS, group * SUFFIX_LEN, CACHE_LEN, device=device)
    context = torch.empty(KV_HEADS, group * SUFFIX_LEN, HEAD_DIM, device=device)
    context_view = context.view(QUERY_HEADS, SUFFIX_LEN, HEAD_DIM)
    weights_view = weights.view(QUERY_HEADS, SUFFIX_LEN, CACHE_LEN)

    def chain_core():
        torch.bmm(grouped, key_t, out=weights)
        pointwise.masked_softmax(weights_view, mask, SCALE)
        torch.bmm(weights, value, out=context)

    def chain_full():
        chain_core()
        pointwise.attention_epilogue(context_view, target)

    def flash():
        pointwise.fused_attention(query, key, value, mask, target, SCALE)

    results = [("empty kernel (one graph node)",
                _median_us(split_attention.noop, reps, inner, device))]
    for precision, enabled in (("fp32", False), ("tf32", True)):
        with with_tf32(enabled):
            results.append((f"cuBLAS {precision} QK + softmax + PV (3 launches)",
                            _median_us(chain_core, reps, inner, device)))
            results.append((f"  + transpose/bf16 epilogue (4 launches)",
                            _median_us(chain_full, reps, inner, device)))
    results.append(("one-warp-per-row flash (1 launch)",
                    _median_us(flash, reps, inner, device)))

    for tile, rows_per_cta, groups, unroll, pieces, tensor in configurations:
        buffers = split_attention.workspace(QUERY_HEADS, SUFFIX_LEN, HEAD_DIM, CACHE_LEN,
                                            device, tile)
        splits = split_attention.splits_for(CACHE_LEN, tile)
        tiles = -(-SUFFIX_LEN // rows_per_cta)

        def slices(tile=tile, rows_per_cta=rows_per_cta, groups=groups, unroll=unroll,
                   tensor=tensor, buffers=buffers):
            split_attention.partials(query, key, value, mask, SCALE, buffers, key_tile=tile,
                                     row_tile=rows_per_cta, row_groups=groups, unroll=unroll,
                                     tensor=tensor)

        def merge(buffers=buffers, pieces=pieces):
            split_attention.combine(buffers, target, pieces=pieces)

        def run(tile=tile, rows_per_cta=rows_per_cta, groups=groups, unroll=unroll,
                pieces=pieces, tensor=tensor, buffers=buffers):
            split_attention.fused_attention(query, key, value, mask, target, SCALE,
                                            buffers=buffers, key_tile=tile,
                                            row_tile=rows_per_cta, row_groups=groups,
                                            unroll=unroll, pieces=pieces, tensor=tensor)

        label = (f"{'tf32' if tensor else 'fp32'} keys={tile} "
                 f"({QUERY_HEADS * splits * tiles} CTAs")
        results.append((f"{label}, slices only)", _median_us(slices, reps, inner, device)))
        results.append((f"{label}, combine only)", _median_us(merge, reps, inner, device)))
        results.append((f"{label}, 2 launches)", _median_us(run, reps, inner, device)))
    return results


def phases(device, groups, tensor=False):
    """Per-CTA SM cycles for each phase, key_tile 40."""
    torch.manual_seed(0)
    query = torch.randn(QUERY_HEADS, SUFFIX_LEN, HEAD_DIM, device=device)
    key = torch.randn(KV_HEADS, CACHE_LEN, HEAD_DIM, device=device)
    value = torch.randn(KV_HEADS, CACHE_LEN, HEAD_DIM, device=device)
    mask = torch.rand(SUFFIX_LEN, CACHE_LEN, device=device) > 0.25
    tile = split_attention.TENSOR_KEY_TILE if tensor else split_attention.DEFAULT_KEY_TILE
    buffers = split_attention.workspace(QUERY_HEADS, SUFFIX_LEN, HEAD_DIM, CACHE_LEN, device,
                                        tile)
    for _ in range(20):
        record = split_attention.timed(query, key, value, mask, SCALE, buffers, groups,
                                       tensor=tensor)
    torch.cuda.synchronize()
    flat = record.view(-1, 4).float()
    return [(name, flat[:, i].median().item(), flat[:, i].max().item())
            for i, name in enumerate(("stage tiles", "QK + softmax", "PV", "whole CTA"))]


def main() -> int:
    parser = argparse.ArgumentParser()
    # "tile:slots" pairs; slots 1 puts 32 query rows in a CTA and 2 puts 64.
    # "keys:rows:groups:unroll:pieces:tensor": the key slice width, the query
    # rows one CTA covers, how its eight warps divide those rows, the QK
    # mainloop's unroll factor, how many warps the combine gives one output row,
    # and whether the TF32 tensor-core mainloops are used (which honour only the
    # key slice width and the combine's warps).
    parser.add_argument("--configurations",
                        default="40:64:1:4:2:0,32:64:1:2:2:1,48:64:1:2:2:1")
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--inner", type=int, default=50)
    parser.add_argument("--prefix", action="store_true",
                        help="also check and time the backbone's prefix shape")
    args = parser.parse_args()
    configurations = [tuple(int(p) for p in item.split(":"))
                      for item in args.configurations.split(",") if item]

    device = "cuda"
    split_attention.build(verbose=True)
    print(f"shape: query[{QUERY_HEADS}, {SUFFIX_LEN}, {HEAD_DIM}] "
          f"cache[{KV_HEADS}, {CACHE_LEN}, {HEAD_DIM}] mask[{SUFFIX_LEN}, {CACHE_LEN}]")
    print(f"touched bytes={BYTES / 1024:.0f} KiB  MACs={MACS / 1e6:.1f} M")
    print("correctness")
    ok = True
    for configuration in configurations:
        ok &= check(device, *configuration)

    print("finite-sentinel path, both mainloops")
    for configuration in configurations[:2]:
        print(f"  -- {'tf32' if configuration[5] else 'fp32'} keys={configuration[0]}")
        ok &= check_sentinel(device, *configuration)

    print("per-CTA SM cycles, keys=40 rows=64 (median / max)")
    cycles = {}
    for label, kwargs in (("fp32 keys=40", {"groups": 1}),
                          ("tf32 keys=48", {"groups": 1, "tensor": True})):
        cycles[label] = phases(device, **kwargs)
        for name, median, worst in cycles[label]:
            print(f"  {label} {name:32s} {median:8.0f} {worst:8.0f}")

    print("latency (median of graph replays, us per layer-step)")
    measured = benchmark(device, configurations, args.reps, args.inner)
    for name, value in measured:
        print(f"  {name:46s} {value:7.2f}")
    for label, record in cycles.items():
        whole = next(c for c in record if c[0] == "whole CTA")[1]
        wall = next((v for n, v in measured if n.startswith(label) and "slices" in n), None)
        if wall:
            # One row tile of 128 CTAs is one wave on 132 SMs, so a CTA's cycles
            # and the launch's wall time price the clock it actually ran at;
            # with two row tiles the ratio is halved by the second wave.
            print(f"  implied SM clock, {label}: {whole / wall / 1e3:.2f} GHz")
    print(f"  {'streaming floor (1.5 MB touched)':46s} {STREAMING_FLOOR_US:7.2f}")
    print(f"  {'float32 FMA roofline (66 M MAC)':46s} {FMA_ROOFLINE_US:7.2f}")

    if args.prefix:
        print("prefix shape (same operation, backbone's 264 rows over 264 keys)")
        ok &= prefix_shape(device, args.reps, args.inner, *configurations[0])

    chain = next(v for n, v in measured if n.startswith("cuBLAS tf32"))
    full = measured[measured.index(next(r for r in measured
                                        if r[0].startswith("cuBLAS tf32"))) + 1][1]
    best_name, best = min(((n, v) for n, v in measured if "2 launches" in n),
                          key=lambda item: item[1])
    print(f"best: {best_name.split(' (')[0]} at {best:.2f} us")
    print(f"  vs the tf32 three-launch chain {chain:.2f} us: {chain / best:.2f}x, "
          f"{(chain - best) * 360 / 1000:.2f} ms per forward")
    print(f"  vs the same output in four {full:.2f} us: {full / best:.2f}x, "
          f"{(full - best) * 360 / 1000:.2f} ms per forward")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
