"""The latency floor model: three columns per call site, all guidance, none a target.

    python -m benchmarks floor --target h100/pi05

Per call site of every stage, at the Target's shapes:

  roofline_us   datasheet: max(bytes / HBM peak, flops / dense bf16 peak), the
                peaks read from the hardware axis's `spec.py`; plan-independent
  ceiling_us    measured: what this machine has delivered for the geometry,
                from the tagged rows of the hardware axis's measured table
                (`measured/constants.yaml`): below the burst curve
                fixed_us + bytes / marginal rate, on the curve bytes /
                delivered rate, against flops / observed tensor rate; a Target
                may declare a call site's ceiling outright (`Invocation.ceiling`
                with its tag and job)
  measured_us   the in-graph time `benchmarks.profile` attributes to the site

and, per site, `pct_of_ceiling` and `within_ceiling`: measured at most
(1 + headroom_pct / 100) times the ceiling, the registry's stop condition
(`eval/acceptance.py`, `stop.headroom_pct`). These comparisons are guidance, not proof that no kernel-level opportunity
remains. Overlapping atomic groups have no supported additive latency model
and cannot trigger automatic stopping.

Validity is checked, not assumed: a site's ceiling above its attributed time
by more than the table's noise floor (`machine.noise_floor_pct`), a stage's
ceiling sum above its measured minimum, or a roofline above its ceiling is a
model error and marks the report invalid. Call sites of one atomic group (a
dependent-launch chain) are judged on the group's sums: under such a chain a
kernel's recorded duration overlaps its neighbours' (`benchmarks.profile`),
so their summed durations are not the chain latency. Under-one-wave launches (grid
below the CTA knee) are counted and reported; their derating
(`ld.ctas.dev.knee`) is carried as information, not applied.

The report carries the measured table's version, so a re-measured constant
changes every floor's version; the datasheet peaks are named with the spec
class they came from.
"""
from __future__ import annotations

from .metrics import report_context

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from eval import acceptance
from flash_vla.runtime.cost import Invocation, total
from flash_vla.runtime.engine import segments

from .latency import _env, _stats, _time_event
from .metrics import require_cuda
from .profile import attribute
from .targets import PLAN_NAMES, build, resolve

#: The model form; bump when a column or a term changes.
FORM_VERSION = "3"
REPO = Path(__file__).resolve().parent.parent
#: hardware axis of the identity -> the package holding `spec.py` and `measured/`.
HARDWARE = {"h100-sxm5-80gb": "flash_vla.hardware.nvidia.h100"}
#: The tagged rows the ceiling column consumes.
TAGS = {
    "stream": "ld.bw.dev.dram",      # TB/s marginal cold streaming rate behind `fixed_us`
    "burst": "tma.bw.dev.burst",     # GB/s delivered end-to-end vs burst size, `curve_mb_gbs`
    "tensor": "wgmma.clock.sm",      # TFLOP/s bf16 observed at real clocks
    "launch": "launch.lat.dev.ramp",  # us of grid ramp per launch (information)
    "knee": "ld.ctas.dev.knee",      # CTAs below which a cold read is derated (information)
}
#: Machine-readable fields a row may carry beside value/units/short/rule.
ROW_FIELDS = ("fixed_us", "curve_mb_gbs", "derate_at_32")


def hardware_axis(hardware: str) -> tuple[type, Path]:
    """The datasheet spec class and the measured table of one hardware axis."""
    package = HARDWARE.get(hardware)
    if package is None:
        raise KeyError(f"no hardware axis known for {hardware!r}; known: {sorted(HARDWARE)}")
    module = importlib.import_module(package)
    spec = importlib.import_module(package + ".spec")
    spec_cls = next(getattr(spec, name) for name in spec.__all__ if name.endswith("Spec"))
    return spec_cls, Path(module.__file__).parent / "measured" / "constants.yaml"


def load_constants(path: Path) -> tuple[dict[str, Any], str]:
    """The tagged rows the ceiling uses (plus the machine's noise floor), and the table's version."""
    import yaml
    raw = path.read_bytes()
    doc = yaml.safe_load(raw)
    rows = {row["tag"]: row for row in doc.get("constants", [])}
    picked: dict[str, Any] = {"noise_floor_pct": float(doc["machine"]["noise_floor_pct"])}
    for role, tag in TAGS.items():
        if tag not in rows:
            raise KeyError(f"constant {tag!r} ({role}) not in {path}")
        row = rows[tag]
        picked[role] = {"tag": tag, "value": row["value"], "units": row.get("units"),
                        "short": row.get("short"),
                        **{f: row[f] for f in ROW_FIELDS if f in row}}
    for role, field in (("stream", "fixed_us"), ("burst", "curve_mb_gbs")):
        if field not in picked[role]:
            raise KeyError(f"row {TAGS[role]!r} in {path} lacks the machine-readable {field!r}")
    return picked, hashlib.sha1(raw).hexdigest()[:12]


def datasheet(spec_cls: type) -> dict[str, Any]:
    """The two datasheet peaks the roofline divides by, with their source named."""
    return {"hbm_bps": float(spec_cls.HBM_BANDWIDTH_BYTES_PER_SECOND),
            "bf16_fps": float(spec_cls.TENSOR_CORE_DENSE_PEAK_FLOPS["bf16"]),
            "source": f"{spec_cls.__module__}.{spec_cls.__name__}"}


def delivered_us(nbytes: int, constants: dict[str, Any]) -> tuple[float, str]:
    """Cold delivery time for `nbytes` from the measured rows, and the rule applied."""
    mb = nbytes / 1e6
    curve = constants["burst"]["curve_mb_gbs"]
    stream = constants["stream"]
    if mb < curve[0][0]:
        return stream["fixed_us"] + mb / float(stream["value"]), (
            f"{stream['fixed_us']} us + MB / {stream['value']} TB/s [{stream['tag']}]")
    for (m0, r0), (m1, r1) in zip(curve, curve[1:]):
        if mb <= m1:
            rate = r0 + (mb - m0) / (m1 - m0) * (r1 - r0)
            return mb * 1e3 / rate, f"MB / {rate:.0f} GB/s interpolated [{constants['burst']['tag']}]"
    return mb * 1e3 / curve[-1][1], f"MB / {curve[-1][1]} GB/s (curve top) [{constants['burst']['tag']}]"


def site_row(invocation: Invocation, peaks: dict[str, Any], constants: dict[str, Any]) -> dict[str, Any]:
    """One call site's roofline and ceiling columns, per call and times its count."""
    cost = invocation.cost
    roof_stream = cost.bytes / peaks["hbm_bps"] * 1e6
    roof_tensor = cost.flops / peaks["bf16_fps"] * 1e6
    roofline = max(roof_stream, roof_tensor)
    tensor_fps = float(constants["tensor"]["value"]) * 1e12
    tensor_us = cost.flops / tensor_fps * 1e6
    if invocation.ceiling is not None:
        ceiling = invocation.ceiling.us
        source = {"declared": True, "tag": invocation.ceiling.tag, "job": invocation.ceiling.job}
    else:
        memory_us, rule = delivered_us(cost.bytes, constants)
        ceiling = max(memory_us, tensor_us)
        source = {"declared": False,
                  "rule": rule if memory_us >= tensor_us else f"FLOPs / {constants['tensor']['value']} TFLOP/s [{constants['tensor']['tag']}]"}
    n = invocation.count
    return {"call_site": invocation.call_site, "count": n,
            "bytes": cost.bytes, "flops": cost.flops,
            "bound": "compute" if roof_tensor > roof_stream else "memory",
            "roofline_us_each": roofline, "roofline_us": roofline * n,
            "ceiling_us_each": ceiling, "ceiling_us": ceiling * n, "ceiling_source": source}


def run(target: str, plan: str | None = None, reps: int = 30, warmup: int = 3, seed: int = 0,
        **overrides) -> dict[str, Any]:
    """Compute the three columns per stage and call site, and check the model against them."""
    require_cuda()
    torch.cuda.init()
    target = resolve(target)
    headroom_pct = acceptance.for_target(target)["stop"]["headroom_pct"]
    engine = build(target, plan or "shipped", seed=seed, **overrides)
    identity = engine.identity
    spec_cls, constants_path = hardware_axis(identity.hardware)
    constants, constants_version = load_constants(constants_path)
    peaks = datasheet(spec_cls)
    limit = 1.0 + headroom_pct / 100.0

    inputs = engine.sample_inputs(seed)
    engine.forward(**inputs)
    torch.cuda.synchronize()
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    costs = engine.costs
    groups = [frozenset(g) for g in engine.atomic_groups]
    tolerance = 1.0 - constants["noise_floor_pct"] / 100.0
    report_segments: dict[str, Any] = {}
    valid = True
    for name in segments(engine):
        rows = [site_row(inv, peaks, constants) for inv in costs.get(name, ())]
        roofline = sum(r["roofline_us"] for r in rows)
        ceiling = sum(r["ceiling_us"] for r in rows)
        profiled = attribute(engine, name, sm_count)
        measured = _stats(_time_event(lambda name=name: engine.replay(name), reps, warmup), reps)
        measured_us = measured["min"] * 1e3
        seg_valid = ceiling <= measured_us and roofline <= ceiling
        attributed = profiled["call_sites"] if profiled["valid"] else {}
        within_all = bool(attributed)
        for row in rows:
            site = attributed.get(row["call_site"])
            row["group"] = next((sorted(g) for g in groups if row["call_site"] in g), None)
            row["measured_us"] = site["dur_us"] if site else None
            row["launches"] = site["launches"] if site else None
            row["under_one_wave"] = site["under_one_wave"] if site else None
            row["pct_of_ceiling"] = (site["dur_us"] / row["ceiling_us"] * 100
                                     if site and row["ceiling_us"] else None)
            row["within_ceiling"] = (site is not None and row["ceiling_us"] > 0
                                     and site["dur_us"] <= limit * row["ceiling_us"])
            row["valid"] = row["roofline_us"] <= row["ceiling_us"] and (
                site is None or row["group"] is not None
                or tolerance * row["ceiling_us"] <= site["dur_us"])
        # No calibrated joint boundary model exists for overlapping atomic groups.
        # Keep their occurrence timings diagnostic and prevent automatic stopping.
        seg_groups = []
        for group in groups:
            members = [r for r in rows if r["call_site"] in group]
            if len(members) < 2:
                continue
            diagnostic = next((g for g in profiled["regions"] if set(g["call_sites"]) == set(group)), None)
            for row in members:
                row["within_ceiling"] = False
                row["pct_of_ceiling"] = None
            seg_groups.append({"call_sites": sorted(group), "ceiling_us": None,
                               "measured_us": None, "pct_of_ceiling": None,
                               "within_ceiling": False, "valid": False,
                               "model_status": "unsupported_overlapping_group",
                               "diagnostic_occurrences": diagnostic})
        if seg_groups:
            seg_valid = False
            within_all = False
        for row in rows:
            seg_valid &= row["valid"]
            within_all &= row["within_ceiling"]
        valid &= seg_valid
        report_segments[name] = {
            "roofline_us": roofline,
            "ceiling_us": ceiling,
            "measured_min_us": measured_us,
            "measured_median_us": measured["median"] * 1e3,
            "pct_of_ceiling": measured_us / ceiling * 100 if ceiling else None,
            "all_within_ceiling": within_all,
            "valid": seg_valid,
            "groups": seg_groups,
            "launches": profiled["launches"],
            "attribution_valid": profiled["valid"],
            "kernel_time_us": profiled["total_us"],
            "copy_us": profiled["copy_us"],
            "under_one_wave_launches": sum(
                s["under_one_wave"] for s in profiled["call_sites"].values()),
            "unattributed_us": profiled["unattributed"]["dur_us"],
            "call_sites": rows,
        }
    totals = {key: sum(seg[key] for seg in report_segments.values())
              for key in ("roofline_us", "ceiling_us", "measured_min_us", "launches")}
    totals["all_within_ceiling"] = all(seg["all_within_ceiling"] for seg in report_segments.values())
    totals["headroom_pct"] = headroom_pct
    report = {
        "identity": identity.as_dict(),
        "measurement_context": report_context(engine, _env()),
        "env": _env(),
        "floor_model": {"form": FORM_VERSION, "constants_file": str(constants_path),
                        "constants_version": constants_version, "constants": constants,
                        "datasheet": peaks,
                        "version": f"{FORM_VERSION}+{constants_version}"},
        "config": {"reps": reps, "warmup": warmup, "seed": seed, "plan": plan},
        "segments": report_segments,
        "declared": total(costs),
        "totals": totals,
        "valid": valid,
        "note": ("guidance, not an objective: roofline is the datasheet, ceiling is what the "
                 "machine delivered for the geometry, measured is the attributed in-graph time; "
                 "a ceiling above a measured time by more than the noise floor, or a roofline "
                 "above a ceiling, invalidates the report; an overlapping dependent-launch chain has "
                 "no supported joint ceiling model; under-one-wave derating is reported, not applied"),
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
