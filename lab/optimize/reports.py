"""Translate existing evaluator reports into Campaign evidence; never run an experiment."""
import argparse
import json
from importlib import import_module
import math
from pathlib import Path

from benchmarks.targets import resolve
from eval import acceptance
from flash_vla.runtime.identity import Identity, inference_signature

from . import measurement, store

OBJECTIVES = {"e2e_chunk_latency_ms": "chunk_latency", "device_latency_ms": "device_latency"}


def compatibility(record, *, context):
    """Bind an observed checkpoint contract to the requested transition assets."""
    if record.get("status") != "passed":
        raise ValueError("checkpoint structural compatibility did not pass")
    identity = Identity.from_dict(record["identity"])
    if inference_signature(**record["contract"]) != identity.inference_signature:
        raise ValueError("observed checkpoint inference signature differs from Target")
    observed = measurement.context(context).as_dict()
    for group, keys in (("weights", ("checkpoint_id", "checkpoint_digest")),
                        ("fixture", ("id", "digest"))):
        if any(record[group][key] != observed[group][key] for key in keys):
            raise ValueError(f"compatibility checked different {group}")
    return dict(record, status="pass", measurement_context=observed)


def latency(report, *, objective, anchor=False, segment=None):
    """Retain real A/B/A legs and enforce the producer's fixed latency-v2 sampling."""
    if report.get("protocol") != "latency-v2":
        raise ValueError("a supported producer benchmark protocol is required")
    if report.get("instrumented") is not False or report["config"].get("attribution") is not False:
        raise ValueError("diagnostic/instrumented timing cannot enter Campaign measurements")
    legs = report["legs"]
    if any(leg.get("attribution") is not None for leg in legs):
        raise ValueError("leg attribution contradicts an uninstrumented measurement")
    if len(legs) != 3:
        raise ValueError("three A/B/A legs are required")
    candidate, parent = legs[1]["identity"], legs[0]["identity"]
    context = legs[1]["measurement_context"]
    policy = acceptance.for_target(candidate["target"])["latency"]
    for key in ("reps", "warmup", "soak_s", "p99_min_reps"):
        if report["config"].get(key) != policy[key]:
            raise ValueError(f"benchmark sampling differs from latency-v2: {key}")
    if report["config"].get("plans") != [leg["plan"] for leg in legs]:
        raise ValueError("reported A/B/A plan order differs from the executed configuration")
    metric = OBJECTIVES[objective]
    for leg in legs:
        for name in {"chunk_latency", metric}:
            stats = leg["metrics"][name]
            if stats["n"] != policy["reps"]:
                raise ValueError("metric sample count differs from the benchmark protocol")
            if any(not isinstance(stats.get(key), (float, int)) or not math.isfinite(stats[key])
                   or stats[key] <= 0 for key in ("min", "median", "p99")):
                raise ValueError("complete finite positive latency statistics are required")
    chunk_controls = [legs[index]["metrics"]["chunk_latency"]["min"]
                      for index in (range(3) if anchor else (0, 2))]
    if max(chunk_controls) - min(chunk_controls) > policy["control_spread_max_ms"]:
        raise ValueError("A/B/A chunk control spread exceeds acceptance policy")
    result = dict(identity=candidate, measurement_context=context, protocol=report["protocol"],
                  validity="valid", instrumented=False, aba=report,
                  parent_incumbent_ms=legs[0]["metrics"][metric]["min"],
                  candidate_ms=legs[1]["metrics"][metric]["min"])
    measurement.comparison(result, expected_identity=candidate, parent_identity=parent,
                           expected_context=context, protocol=report["protocol"], objective=objective)
    if anchor:
        for leg in legs:
            measurement.identity(leg["identity"], candidate)
        result["objective"] = dict(name=objective, unit="ms", value=result.pop("candidate_ms"))
        del result["parent_incumbent_ms"]
    else:
        if not isinstance(segment, int) or isinstance(segment, bool) or segment < 0:
            raise ValueError("candidate evidence needs the active measurement segment")
        result["measurement_segment"] = dict(id=segment, benchmark_protocol=report["protocol"],
                                            **context["environment"])
    return result


def _in_engine(check, expected, configuration):
    report = check["report"]
    if report["config"].get("oracle") != "in_engine_reference":
        raise ValueError("in-engine evidence must name its numerical oracle")
    shape = dict(expected["shape"])
    for name, value in configuration.get("config", {}).items():
        wanted = shape[name] if value == "full" else value
        if report["config"][name] != wanted:
            raise ValueError("correctness ladder depth differs from acceptance")
        shape[name] = wanted
    # Shallow graphs can omit call sites present in the full workload.
    # Resolve against the registered graph, never keys supplied by the report.
    target = import_module("flash_vla." + resolve(expected["target"]).replace("/", ".")).TARGET
    plan = target.registry.resolve(expected["plan"], target.graph(shape).call_sites)
    identity = dict(expected, shape=shape, plan=plan)
    measurement.identity(report["identity"]["candidate"], identity)
    oracle = report["numerical_oracle"]["identity"]
    measurement.identity(oracle, dict(identity, plan=oracle["plan"]))
    measurement.identity(report["identity"]["reference"], oracle)
    reference = measurement.context(report["measurement_context"]["reference"])
    candidate = measurement.context(report["measurement_context"]["candidate"])
    if reference.segment_key != candidate.segment_key:
        raise ValueError("correctness reference checkpoint/fixture/environment differs from candidate")
    if report["replay_identical"] is not True or report["finite"] is not True:
        raise ValueError("correctness requires finite, replay-identical outputs at every depth")
    if configuration["mode"] == "gate":
        key = configuration["threshold"]
        tolerance = acceptance.tolerances(Identity.from_dict(expected).precision)[key]
        if (report.get("threshold_key") != key or report["tolerance"] != tolerance
                or not report["min_cosine"] > tolerance["cosine_min"]
                or not report["max_rel_rms"] < tolerance["rel_rms_max"]):
            raise ValueError("required numerical correctness tolerance did not pass")
    return candidate


def correctness(record):
    """Use the registered ladder, not a process exit code or a report-only 'passed'."""
    expected = record["identity"]
    measurement.identity(expected, expected)
    variant = expected["execution_variant"]
    if variant["quantization"]["mode"] != "bf16" or variant["cache"]["mode"] != "none":
        raise ValueError("nondefault ExecutionVariant requires a supported policy-quality contract")
    target_policy = acceptance.for_target(expected["target"])
    policy = target_policy["correctness"]
    checks = {item["check"]: item for item in record["checks"]}
    if len(checks) != len(record["checks"]):
        raise ValueError("duplicate correctness checks")
    contexts = []
    for configuration in policy["checks"]:
        name = configuration["check"]
        if name not in checks:
            raise ValueError(f"missing required ladder evidence: {name}")
        check = checks[name]
        if check["mode"] != configuration["mode"]:
            raise ValueError(f"correctness mode differs from acceptance: {name}")
        if configuration["mode"] == "gate" and check["status"] != "passed":
            raise ValueError(f"required correctness check did not pass: {name}")
        if configuration.get("oracle") == "in_engine_reference":
            contexts.append(_in_engine(check, expected, configuration))
    if not contexts or any(item.segment_key != contexts[0].segment_key for item in contexts):
        raise ValueError("correctness ladder changed checkpoint/fixture/environment")
    observed = contexts[0].as_dict()
    official_sources = []
    # Official adapter evidence certifies the reference/weights relation. Its
    # own engine revision and environment are not the candidate implementation.
    for configuration in policy["checks"]:
        if configuration.get("oracle") != "official_baseline" or configuration["mode"] != "gate":
            continue
        scripts = checks[configuration["check"]]["scripts"]
        names = [script.get("script") for script in scripts]
        expected_scripts = list(target_policy["scripts"]["official_baseline"])
        if not scripts or sorted(names, key=str) != sorted(expected_scripts):
            raise ValueError("official correctness evidence must cover the registered adapter scripts")
        for script in scripts:
            if script["status"] != "passed" or script["returncode"] != 0 or not script["identities"]:
                raise ValueError("official correctness adapter did not pass")
            if len(script["identities"]) != len(script["weights"]):
                raise ValueError("official adapter provenance is incomplete")
            acceptance.validate_baseline_workloads(
                script["script"], script["identities"], Identity.from_dict(expected),
                script.get("stages", ()))
            provenance = script.get("reference_provenance", [])
            official_sources.extend((item.get("repository"), item.get("commit"))
                                    for item in provenance or [{}])
            if any(provenance):
                observed["reference_provenance"].setdefault("official_baselines", {})[script["script"]] = provenance
            for weights in script["weights"]:
                if any(weights.get(key) != observed["weights"][key]
                       for key in ("checkpoint_id", "checkpoint_digest")):
                    raise ValueError("official correctness adapter checked another checkpoint")
    if any(repository or commit for repository, commit in official_sources):
        observed["reference_provenance"].pop("repository", None)
        observed["reference_provenance"].pop("commit", None)
    if official_sources and all(source == official_sources[0] for source in official_sources):
        repository, commit = official_sources[0]
        if repository and commit:
            observed["reference_provenance"].update(repository=repository, commit=commit)
    return dict(status="pass", identity=expected, measurement_context=observed,
                checks=[dict(check=c["check"], mode=c["mode"], status=c["status"]) for c in record["checks"]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("compatibility", "correctness", "anchor", "comparison"))
    parser.add_argument("report", type=Path)
    parser.add_argument("--objective", choices=tuple(OBJECTIVES), default="e2e_chunk_latency_ms")
    parser.add_argument("--segment", type=int)
    parser.add_argument("--correctness", type=Path, help="normalized correctness evidence for an anchor")
    parser.add_argument("--context", type=Path, help="requested measurement context for compatibility")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    record = store.read(args.report)
    if args.kind == "anchor" and "legs" in record and args.correctness is None:
        parser.error("a raw latency anchor requires --correctness with normalized evidence")
    if args.kind == "compatibility":
        if args.context is None:
            parser.error("compatibility requires --context")
        result = compatibility(record, context=store.read(args.context))
    elif args.kind == "correctness":
        result = correctness(record)
    else:
        raw = record if "legs" in record else record["latency"]["report"]
        result = latency(raw, objective=args.objective, anchor=args.kind == "anchor", segment=args.segment)
        if args.kind == "anchor":
            checked = store.read(args.correctness) if args.correctness else correctness(record)
            measurement.correctness(checked, result["identity"], result["measurement_context"])
            result["correctness"] = checked
    result["evidence"] = str(args.report.resolve())
    if args.out:
        store.write(args.out, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
