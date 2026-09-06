"""The latency floor model: the derived objective of a Target.

    python -m benchmarks floor --target h100/pi05

Two tiers per segment, every term dividing by a measured, tagged constant of
the `hardware-unit-test` skill:

  roofline     per call site, max(bytes / streaming rate, flops / tensor rate),
               times its invocation count; plan-independent, from the
               Target's cost declarations
  structural   roofline plus the launch count of the captured segment times
               the measured grid ramp; plan-dependent, the launch count read
               from a profiler trace of one replay

and the gap decomposition against the measured segment minimum:

  measured - structural   what kernel-level work can still recover
  structural - roofline   what a change of the plan's form can recover

The model is validated in the same report, per segment and per call site: a
structural floor above the measured segment time, or a call site's roofline
above its attributed in-graph time (from `benchmarks.profile`), is a model
error and marks the report invalid. Under-one-wave kernels (grid below the
CTA knee) are counted and reported; their derating is not yet a term of the
structural tier, so the structural floor is optimistic where they dominate.

Datasheet peaks appear nowhere. The report carries the constants file's
version so a re-measured constant changes every floor's version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from flash_vla.runtime.cost import Invocation, total
from flash_vla.runtime.engine import segments

from .latency import _env, _stats, _time_event
from .metrics import require_cuda
from .profile import attribute
from .targets import PLAN_NAMES, build, resolve

#: The model form; bump when a term is added or changed.
FORM_VERSION = "1"
REPO = Path(__file__).resolve().parent.parent
CONSTANTS_ROOT = REPO / ".claude" / "skills" / "hardware-unit-test"
#: hardware axis of the identity -> constants directory of the skill.
ARCH = {"h100-sxm5-80gb": "sm90"}
#: The tags the form consumes.
TAGS = {
    "stream_tbps": "ld.bw.dev.dram",       # TB/s marginal streaming rate, cold
    "tensor_tflops": "wgmma.clock.sm",     # TFLOP/s bf16 observed at real clocks
    "launch_us": "launch.lat.dev.ramp",    # us of grid ramp per launch
    "cta_knee": "ld.ctas.dev.knee",        # CTAs below which a cold read is derated
}


def load_constants(hardware: str) -> tuple[dict[str, Any], str, str]:
    """The tagged rows the form uses, the constants file path and its version."""
    import yaml
    arch = ARCH.get(hardware)
    if arch is None:
        raise KeyError(f"no constants directory known for hardware {hardware!r}; "
                       f"known: {sorted(ARCH)}")
    path = CONSTANTS_ROOT / arch / "constants.yaml"
    raw = path.read_bytes()
    doc = yaml.safe_load(raw)
    rows = {row["tag"]: row for row in doc.get("constants", doc if isinstance(doc, list) else [])}
    if not rows:
        for section in doc.values() if isinstance(doc, dict) else []:
            if isinstance(section, list):
                rows.update({r["tag"]: r for r in section if isinstance(r, dict) and "tag" in r})
    picked = {}
    for role, tag in TAGS.items():
        if tag not in rows:
            raise KeyError(f"constant {tag!r} ({role}) not in {path}")
        row = rows[tag]
        picked[role] = {"tag": tag, "value": row["value"], "units": row.get("units"),
                        "short": row.get("short")}
    return picked, str(path), hashlib.sha1(raw).hexdigest()[:12]


def roofline_us(invocation: Invocation, stream_bps: float, tensor_fps: float) -> dict[str, Any]:
    """One invocation's roofline: the larger of its streaming and tensor times."""
    cost = invocation.cost
    stream = cost.bytes / stream_bps * 1e6
    tensor = cost.flops / tensor_fps * 1e6
    each = max(stream, tensor)
    return {"call_site": invocation.call_site, "count": invocation.count,
            "bytes": cost.bytes, "flops": cost.flops,
            "bound": "compute" if tensor > stream else "memory",
            "roofline_us_each": each, "roofline_us": each * invocation.count}


def run(target: str, plan: str | None = None, reps: int = 30, warmup: int = 3, seed: int = 0,
        **overrides) -> dict[str, Any]:
    """Compute both floor tiers per segment, measure the segment, and decompose the gap."""
    require_cuda()
    torch.cuda.init()
    target = resolve(target)
    engine = build(target, plan or "shipped", seed=seed, **overrides)
    identity = engine.identity
    constants, constants_path, constants_version = load_constants(identity.hardware)
    stream_bps = float(constants["stream_tbps"]["value"]) * 1e12
    tensor_fps = float(constants["tensor_tflops"]["value"]) * 1e12
    launch_us = float(constants["launch_us"]["value"])

    inputs = engine.sample_inputs(seed)
    engine.forward(**inputs)
    torch.cuda.synchronize()
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    costs = engine.costs
    report_segments: dict[str, Any] = {}
    valid = True
    for name in segments(engine):
        rows = [roofline_us(inv, stream_bps, tensor_fps) for inv in costs.get(name, ())]
        roofline = sum(r["roofline_us"] for r in rows)
        profiled = attribute(engine, name, sm_count)
        launches = profiled["launches"]
        structural = roofline + launches * launch_us
        measured = _stats(_time_event(lambda name=name: engine.replay(name), reps, warmup), reps)
        measured_us = measured["min"] * 1e3
        seg_valid = structural <= measured_us
        # Per call site: the attributed in-graph time must not sit below its
        # own roofline; an unattributed segment reports no per-site check.
        attributed = profiled["call_sites"] if profiled["valid"] else {}
        for row in rows:
            site = attributed.get(row["call_site"])
            row["measured_us"] = site["dur_us"] if site else None
            row["launches"] = site["launches"] if site else None
            row["under_one_wave"] = site["under_one_wave"] if site else None
            row["valid"] = (site is None) or (row["roofline_us"] <= site["dur_us"])
            row["above_roofline"] = (site["dur_us"] / row["roofline_us"]
                                     if site and row["roofline_us"] else None)
            seg_valid &= row["valid"]
        valid &= seg_valid
        report_segments[name] = {
            "roofline_us": roofline,
            "structural_us": structural,
            "measured_min_us": measured_us,
            "measured_median_us": measured["median"] * 1e3,
            "gap_kernel_us": measured_us - structural,
            "gap_form_us": structural - roofline,
            "above_structural": measured_us / structural if structural else None,
            "above_roofline": measured_us / roofline if roofline else None,
            "valid": seg_valid,
            "launches": launches,
            "attribution_valid": profiled["valid"],
            "kernel_time_us": profiled["total_us"],
            "copy_us": profiled["copy_us"],
            "under_one_wave_launches": sum(
                s["under_one_wave"] for s in profiled["call_sites"].values()),
            "unattributed_us": profiled["unattributed"]["dur_us"],
            "call_sites": rows,
        }
    sums = total(costs)
    totals = {key: sum(seg[key] for seg in report_segments.values())
              for key in ("roofline_us", "structural_us", "measured_min_us",
                          "gap_kernel_us", "gap_form_us", "launches")}
    report = {
        "identity": identity.as_dict(),
        "env": _env(),
        "floor_model": {"form": FORM_VERSION, "constants_file": constants_path,
                        "constants_version": constants_version, "constants": constants,
                        "version": f"{FORM_VERSION}+{constants_version}"},
        "config": {"reps": reps, "warmup": warmup, "seed": seed, "plan": plan},
        "segments": report_segments,
        "declared": sums,
        "totals": totals,
        "valid": valid,
        "note": ("under-one-wave derating is reported, not modelled; a structural floor "
                 "above a measured segment time, or a call-site roofline above its "
                 "attributed in-graph time, invalidates the report"),
    }
    del engine
    torch.cuda.empty_cache()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True)
    parser.add_argument("--plan", default=None,
                        help=f"one of {PLAN_NAMES}, a JSON object or a lab/plans/*.json path")
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    from .latency import parse_options
    report = run(args.target, args.plan, reps=args.reps, seed=args.seed,
                 **parse_options(args.option))
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
