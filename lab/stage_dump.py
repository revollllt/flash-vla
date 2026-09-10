"""Dump a Target's stage outputs on one plan, and compare two dumps bit for bit.

    python -m lab.stage_dump dump --target h100/pi05 --plan shipped --out a.pt
    python -m lab.stage_dump compare a.pt b.pt

A dump holds every declared stage output (`engine.stage_outputs`) after one
full forward at the given seed, the output of a second forward, the identity
and the git revision of the tree that produced it. `compare` is the
cross-revision bit-identity check the promotion PRs use when a kernel moves
or a wrapper changes without a numerical change: run `dump` from the tree
before and the tree after on one node with one seed, then compare on any
node. Identity is `torch.equal`, so any difference is a finding; the error
metrics of a non-identical pair come from `python -m tools.calibrate --dumps`,
which reads the same file format.

Run from a tree other than the one holding this file with
`PYTHONPATH=<tree>/src:<tree> python <this file> dump ...`, so the harness
factories and the runtime come from that tree.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def _revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # not a git tree
        return "unknown"


def dump(target: str, plan: str, out: str, seed: int = 0, steps: int | None = None,
         layers: int | None = None) -> dict[str, Any]:
    from flash_vla.inference import build, resolve

    depth = {k: v for k, v in (("steps", steps), ("layers", layers)) if v is not None}
    engine = build(resolve(target), plan, seed=seed, **depth)
    inputs = engine.sample_inputs(seed)
    engine.forward(**inputs)
    torch.cuda.synchronize()
    stage = {f"{name}/{buffer}": engine.buffers[buffer].detach().clone().cpu()
             for name, outputs in engine.stage_outputs.items() for buffer, _ in outputs}
    second = engine.forward(**inputs).detach().clone().cpu()
    torch.cuda.synchronize()
    record = {"stage": stage, "second": second, "identity": engine.identity.as_dict(),
              "revision": _revision(), "seed": seed}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, out)
    return record


def compare(a: str, b: str) -> bool:
    x, y = torch.load(a, map_location="cpu"), torch.load(b, map_location="cpu")
    same = True
    keys = sorted(set(x["stage"]) | set(y["stage"]))
    for key in keys:
        if key not in x["stage"] or key not in y["stage"]:
            print(f"  {key}: only in {'A' if key in x['stage'] else 'B'}")
            same = False
            continue
        p, q = x["stage"][key], y["stage"][key]
        equal = p.shape == q.shape and p.dtype == q.dtype and bool(torch.equal(p, q))
        if not equal and p.shape == q.shape:
            diff = (p.float() - q.float()).abs()
            detail = f"max |diff| {diff.max().item():.3e} at {int((diff > 0).sum())} elements"
        else:
            detail = f"{tuple(p.shape)} {p.dtype} vs {tuple(q.shape)} {q.dtype}"
        print(f"  {key}: {'identical' if equal else 'DIFFERENT ' + detail}")
        same &= equal
    out_equal = bool(torch.equal(x["second"], y["second"]))
    print(f"  output (second forward): {'identical' if out_equal else 'DIFFERENT'}")
    same &= out_equal
    print(f"revisions {x.get('revision')} vs {y.get('revision')}, seeds {x.get('seed')} / "
          f"{y.get('seed')}: {'BIT-IDENTICAL' if same else 'NOT IDENTICAL'}")
    return same


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump", help="run one forward and save the stage outputs")
    d.add_argument("--target", required=True)
    d.add_argument("--plan", default="shipped")
    d.add_argument("--out", required=True)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--steps", type=int, default=None)
    d.add_argument("--layers", type=int, default=None)
    c = sub.add_parser("compare", help="compare two dumps bit for bit")
    c.add_argument("a")
    c.add_argument("b")
    args = parser.parse_args(argv)
    if args.command == "dump":
        record = dump(args.target, args.plan, args.out, args.seed, args.steps, args.layers)
        print(f"wrote {args.out}: {len(record['stage'])} stage outputs, revision "
              f"{record['revision']}, plan {sorted(set(record['identity']['plan'].values()))}")
        return 0
    return 0 if compare(args.a, args.b) else 1


if __name__ == "__main__":
    sys.exit(main())
