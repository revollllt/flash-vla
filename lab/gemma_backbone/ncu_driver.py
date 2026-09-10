"""Invoke one call site of one Target a fixed number of times, for a profiler.

    python -m lab.gemma_backbone.ncu_driver --target h100/pi05 \
        --segment llm_backbone --site llm_backbone_norm_gated_ffn --calls 1

Nsight Compute needs a driver that issues exactly the launches to be captured
and nothing else, so `--launch-count` means what it says. This builds the
engine on a plan, records one eager run of the segment the way
`benchmarks/kernels.py` does -- so the arguments are exactly what the pipeline
passes -- and then replays the recorded invocations of one call site.

Warmup runs before the marker so kernel compilation and any tensor-map
encoding are outside the captured region; `--calls` launches follow. The
number of CUDA kernels per call is a property of the routed wrapper, not of
this script (the TileLang gated FFN is two, the CUDA attention is one), so the
caller sets `--launch-count` from the route it is profiling.
"""
from __future__ import annotations

import argparse

import torch

from benchmarks.kernels import record_invocations
from flash_vla.inference import build, resolve


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--target", required=True)
    parser.add_argument("--plan", default="shipped")
    parser.add_argument("--segment", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--calls", type=int, default=1, help="invocations after warmup")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run this on a GPU node")

    engine = build(resolve(args.target), args.plan, seed=args.seed)
    engine.forward(**engine.sample_inputs(args.seed))
    torch.cuda.synchronize()

    calls = record_invocations(engine, args.segment)
    if args.site not in calls:
        raise SystemExit(f"{args.site!r} was not invoked in segment {args.segment!r}; "
                         f"recorded: {sorted(calls)}")
    fn = getattr(engine.ops, args.site)
    recorded = calls[args.site]
    print(f"[driver] {args.site}: {len(recorded)} recorded invocations, "
          f"warmup {args.warmup}, capturing {args.calls}", flush=True)

    for i in range(args.warmup):
        call_args, call_kwargs = recorded[i % len(recorded)]
        fn(*call_args, **call_kwargs)
    torch.cuda.synchronize()

    for i in range(args.calls):
        call_args, call_kwargs = recorded[i % len(recorded)]
        fn(*call_args, **call_kwargs)
    torch.cuda.synchronize()
    print("[driver] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
