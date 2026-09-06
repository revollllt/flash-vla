"""Declaration check of every Target, without a device: python -m eval.smoke.

Builds each Target's graph through the runner's declaration path (no
checkpoint, no capture, no CUDA) and checks what can be checked on a login
node: the identity's shape keys, that every node's call site has a spec and
the right argument count, that every output and every weight reference is
declared, that the canonical stage outputs exist with the exposed shape their
declaration implies, that the graph's derived costs are non-zero per stage,
that the shipped and reference plans and every candidate plan under
`lab/plans/` route every call site, and -- for Pi0.5 -- that the binding
rules accept exactly the route combinations they accepted before.

Every check that can falsify a kernel needs a GPU and lives elsewhere; this
is the cheapest thing that catches a graph that cannot run.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from benchmarks.targets import TARGETS, declare
from flash_vla.runtime.graph import BufRef, WeightRef
from flash_vla.runtime.vla import STAGE_OUTPUTS

REPO = Path(__file__).resolve().parent.parent
SHAPE_KEYS = ("num_views", "chunk", "steps", "layers", "prompt_len")

#: Pi0.5's five plan-selectable action-expert call sites and the backends a
#: route may name; the accepted set is fixed by the CUDA backend's constraints.
_PI05_SITES = ("action_expert_norm_qkv_rope", "action_expert_attention",
               "action_expert_out_proj_residual", "action_expert_norm_gated_ffn",
               "action_expert_ffn_down_residual")
_PI05_BACKENDS = ("tilelang", "cuda", "cuda-pdl")


def _check(results: list[dict[str, Any]], name: str, ok: bool, detail: Any = None) -> None:
    results.append({"check": name, "passed": bool(ok), "detail": detail})


def check_target(target: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    runner = declare(target)
    graph = runner.graph
    _check(results, "identity.shape keys", tuple(runner.identity.shape) == SHAPE_KEYS,
           tuple(runner.identity.shape))
    _check(results, "stages non-empty",
           all(graph.nodes_of(s) for s in graph.segment_names), graph.segment_names)
    bad_arity = [n.index for n in graph.nodes
                 if not n.is_copy and len(n.args) != len(graph.vocabulary[n.call_site].params)]
    _check(results, "node arity matches spec", not bad_arity, bad_arity[:5])
    undeclared = [n.index for n in graph.nodes for r in n.refs()
                  if (isinstance(r, BufRef) and r.name not in graph.buffers)
                  or (isinstance(r, WeightRef) and r.name not in graph.weight_shapes)]
    _check(results, "references declared", not undeclared, undeclared[:5])
    outputs_ok = True
    for stage, outputs in STAGE_OUTPUTS.items():
        for buffer, axis in outputs:
            spec = graph.buffers.get(buffer)
            if spec is None:
                outputs_ok = False
                continue
            base = graph.buffers[spec.alias] if spec.alias else spec
            exposed = graph.buf(buffer).shape
            if axis is not None and (axis >= len(exposed) or exposed[axis] < 1):
                outputs_ok = False
    _check(results, "stage outputs declared", outputs_ok)
    costs = runner.costs
    _check(results, "costs non-zero per stage",
           all(sum(i.bytes for i in rows) > 0 and sum(i.flops for i in rows) > 0
               for rows in costs.values()),
           {s: len(rows) for s, rows in costs.items()})
    for plan in ("shipped", "reference"):
        routed = declare(target, plan).identity.plan
        _check(results, f"plan {plan} routes every call site",
               set(routed) == set(graph.call_sites), sorted(set(routed.values())))
    return results


def check_lab_plans() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted((REPO / "lab" / "plans").glob("*.json")):
        target = "h100/" + path.stem.split("-")[0]
        try:
            runner = declare(target, str(path))
            ok = set(runner.identity.plan) == set(runner.graph.call_sites)
            detail = sorted(set(runner.identity.plan.values()))
        except Exception as exc:  # a plan that cannot bind is the finding
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        _check(results, f"lab plan {path.name}", ok, detail)
    return results


def check_pi05_routes() -> list[dict[str, Any]]:
    """Every 3^5 route of the five action-expert sites: accepted iff the constraints allow."""
    from flash_vla.hardware.nvidia.h100.pi05 import TARGET

    accepted, rejected = [], []
    call_sites = declare("h100/pi05").graph.call_sites
    for combo in itertools.product(_PI05_BACKENDS, repeat=len(_PI05_SITES)):
        plan = dict(zip(_PI05_SITES, combo))
        try:
            TARGET.registry.resolve(plan, call_sites)
            accepted.append(combo)
        except ValueError as exc:
            rejected.append((combo, str(exc)))

    def expected(combo) -> bool:
        qkv, attn, oproj, gate, down = combo
        cuda = {"cuda", "cuda-pdl"}
        # The attention pair moves together; the FFN pair moves together; the
        # fused out-projection needs the FFN pair on its own backend.
        pair_attn = (qkv == attn) or (qkv not in cuda and attn not in cuda)
        pair_ffn = (gate == down) or (gate not in cuda and down not in cuda)
        oproj_ok = oproj not in cuda or (gate == oproj and down == oproj)
        return pair_attn and pair_ffn and oproj_ok

    mismatches = [c for c in itertools.product(_PI05_BACKENDS, repeat=5)
                  if (c in accepted) != expected(c)]
    named = all("requires" in msg for _, msg in rejected)
    return [{"check": "pi05 route combinations", "passed": not mismatches and named,
             "detail": {"accepted": len(accepted), "rejected": len(rejected),
                        "mismatches": mismatches[:5]}}]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", action="store_true", help="print the full result list")
    args = parser.parse_args(argv)
    report: dict[str, list[dict[str, Any]]] = {}
    for target in TARGETS:
        report[target] = check_target(target)
    report["lab/plans"] = check_lab_plans()
    report["pi05 routes"] = check_pi05_routes()
    failed = [(k, r["check"], r["detail"]) for k, rows in report.items()
              for r in rows if not r["passed"]]
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        for k, rows in report.items():
            print(f"{k}: {sum(r['passed'] for r in rows)}/{len(rows)} passed")
        for k, name, detail in failed:
            print(f"  FAIL {k}: {name}: {detail}")
    print("PASS" if not failed else "FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
