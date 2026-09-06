"""Declaration check of every Target, without a device: python -m eval.smoke.

Builds each Target's graph through the runner's declaration path (no
checkpoint, no capture, no CUDA) and checks what can be checked on a login
node: the identity's shape keys, that every node's call site has a spec and
the right argument count, that every output and every weight reference is
declared, that the canonical stage outputs exist with the exposed shape their
declaration implies, that the graph's derived costs are non-zero per stage,
that the shipped and reference plans and every candidate plan under
`lab/plans/` route every call site (a candidate's Target is read from its
file-name prefix, `lab/plans/README.md`), and that each Target's binding
rules accept exactly the route combinations the oracle here expects of them.

Every check that can falsify a kernel needs a GPU and lives elsewhere; this
is the cheapest thing that catches a graph that cannot run.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from benchmarks.targets import TARGETS, declare, resolve
from flash_vla.runtime.graph import BufRef, WeightRef
from flash_vla.runtime.vla import STAGE_OUTPUTS

REPO = Path(__file__).resolve().parent.parent
SHAPE_KEYS = ("num_views", "chunk", "steps", "layers", "prompt_len")


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
    """Every `lab/plans/<target>-<name>.json` binds on the Target its prefix names."""
    results: list[dict[str, Any]] = []
    for path in sorted((REPO / "lab" / "plans").glob("*.json")):
        prefix = path.stem.split("-")[0]
        try:
            target = resolve("h100/" + prefix)
        except KeyError:
            _check(results, f"lab plan {path.name}", False,
                   f"prefix {prefix!r} names no Target (lab/plans/README.md)")
            continue
        try:
            runner = declare(target, str(path))
            ok = set(runner.identity.plan) == set(runner.graph.call_sites)
            detail = sorted(set(runner.identity.plan.values()))
        except Exception as exc:  # a plan that cannot bind is the finding
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        _check(results, f"lab plan {path.name}", ok, detail)
    return results


# --- route oracles ----------------------------------------------------------
# Per Target, the hand-written expectation of which route combinations bind.
# Written independently of the backends' `ROUTE_CONSTRAINTS` on purpose: a
# constraint that drifts from the documented buffer contract is caught by
# disagreeing with the prose rule below. A Target whose backends declare any
# constraint must have an oracle here; a backend without constraints needs
# none (every combination binds).

def _pi05_routes(plan: Mapping[str, str]) -> bool:
    """The CUDA backend's contracts: the decoder attention pair moves together,
    the FFN pair moves together, and the fused out-projection needs the FFN
    pair on its own backend. Every other call site routes freely."""
    cuda = {"cuda", "cuda-pdl"}
    qkv = plan.get("action_expert_norm_qkv_rope")
    attn = plan.get("action_expert_attention")
    oproj = plan.get("action_expert_out_proj_residual")
    gate = plan.get("action_expert_norm_gated_ffn")
    down = plan.get("action_expert_ffn_down_residual")
    pair_attn = (qkv == attn) or (qkv not in cuda and attn not in cuda)
    pair_ffn = (gate == down) or (gate not in cuda and down not in cuda)
    oproj_ok = oproj not in cuda or (gate == oproj and down == oproj)
    return pair_attn and pair_ffn and oproj_ok


_ROUTE_ORACLES: dict[str, Callable[[Mapping[str, str]], bool]] = {
    "hardware/nvidia/h100/pi05": _pi05_routes,
    "hardware/nvidia/h100/pi0": lambda plan: True,
}


def check_routes(target: str) -> list[dict[str, Any]]:
    """Every route combination over the constrained call sites, and every
    plan-selectable call site alone on each backend that provides it:
    accepted iff the Target's oracle says so, and every rejection is named."""
    target = resolve(target)
    runner = declare(target)
    registry = runner.target.registry
    provided = registry.provided()
    call_sites = list(runner.graph.call_sites)
    constrained = set()
    for declared in registry.constraints().values():
        for constraint in declared:
            constrained |= set(constraint.members)
    selectable = [s for s in call_sites
                  if sum(s in names for names in provided.values()) > 1]
    oracle = _ROUTE_ORACLES.get(target)
    if constrained and oracle is None:
        return [{"check": f"{target} route combinations", "passed": False,
                 "detail": "backends declare route constraints but eval/smoke.py has no "
                           "route oracle for this Target"}]
    oracle = oracle or (lambda plan: True)

    def bind(plan: dict[str, str]):
        try:
            registry.resolve(plan, call_sites)
            return True, None
        except ValueError as exc:
            return False, str(exc)

    trials: list[tuple[dict[str, str], bool, str | None]] = []
    cluster = [s for s in selectable if s in constrained]
    choices = [sorted(b for b, names in provided.items() if s in names) for s in cluster]
    for combo in itertools.product(*choices):
        plan = dict(zip(cluster, combo))
        ok, message = bind(plan)
        trials.append((plan, ok, message))
    for site in selectable:
        if site in cluster:
            continue
        for backend, names in provided.items():
            if site in names:
                plan = {site: backend}
                ok, message = bind(plan)
                trials.append((plan, ok, message))
    mismatches = [plan for plan, ok, _ in trials if ok != oracle(plan)]
    rejected = [(plan, message) for plan, ok, message in trials if not ok]
    named = all(message is not None and "requires" in message for _, message in rejected)
    return [{"check": f"{target} route combinations", "passed": not mismatches and named,
             "detail": {"backends": sorted(provided), "cluster": cluster,
                        "trials": len(trials), "accepted": len(trials) - len(rejected),
                        "rejected": len(rejected), "mismatches": mismatches[:5]}}]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", action="store_true", help="print the full result list")
    args = parser.parse_args(argv)
    report: dict[str, list[dict[str, Any]]] = {}
    for target in TARGETS:
        report[target] = check_target(target)
    report["lab/plans"] = check_lab_plans()
    for target in TARGETS:
        report[f"{target} routes"] = check_routes(target)
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
