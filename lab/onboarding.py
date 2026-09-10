"""Persistent, model-agnostic evidence for onboarding a new Target."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time

from lab.optimize import store, campaign, measurement, transition
from lab.optimize.registry import CampaignRegistry, key_digest
from flash_vla.runtime.identity import ExecutionVariant, inference_signature


LEGACY_STAGES = (
    "requirement_freeze",
    "upstream_freeze",
    "official_reference",
    "compatibility_scan",
    "target_bring_up",
    "correctness_ladder",
    "baseline_ladder",
    "floor_profile",
    "campaign_creation",
)

STAGES = (
    "requirement_freeze", "reference_freeze", "model_contract", "weights_compatibility",
    "official_reference", "compatibility_scan", "target_bring_up",
    "correctness_ladder", "baseline_ladder", "floor_profile", "campaign_creation", "publication",
)


def _stages(spec):
    return STAGES if spec["version"] == 2 else LEGACY_STAGES


COMPATIBILITY_CATEGORIES = (
    "existing runtime op",
    "existing shared component",
    "Target-local composition",
    "Target-local op",
    "new reusable component",
    "genuine runtime primitive",
    "host work",
    "capture blocker",
    "unsupported",
)

COMPATIBILITY_RISKS = (
    "dynamic shape",
    "post-freeze allocation",
    "host/device sync",
    "graph capture blocker",
    "data-dependent control flow",
)

CORRECTNESS_LADDER = (
    "declaration smoke",
    "op/component parity",
    "stage parity",
    "shallow model",
    "full layers",
    "single denoise step",
    "full denoise loop",
    "official end-to-end parity",
)

BASELINE_LADDER = (
    "upstream eager",
    "upstream official optimized/compile",
    "flash-vla reference plan",
    "flash-vla initial shipped plan",
)

OFFICIAL_REFERENCE_CHECKS = (
    "upstream_load",
    "checkpoint",
    "tokenizer",
    "processor",
    "correctness_fixture",
    "performance_fixture",
    "seed_noise",
    "upstream_eager",
    "upstream_official_optimized",
    "reference_outputs",
    "upstream_environment",
)

TARGET_BRING_UP_CHECKS = (
    "model_schema",
    "weight_loader",
    "tokenizer_processor_contract",
    "target",
    "pipeline",
    "reference_plan",
    "shipped_plan",
    "registry",
    "backend_implementation",
    "benchmark_registration",
    "acceptance_registration",
    "official_baseline_adapter",
    "smoke",
)

RUNTIME_MODIFICATION_CHECKS = (
    "model_agnostic",
    "pi0_smoke",
    "pi05_smoke",
    "dependency_direction",
    "benchmark_api",
)


def _mapping(value, name):
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty object")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _list(value, name, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise ValueError(f"{name} must be a {qualifier}list")
    return value


def _check_result(value, name, *, unavailable=False):
    value = _mapping(value, name)
    allowed = {"passed"} | ({"unavailable"} if unavailable else set())
    if value.get("status") not in allowed:
        raise ValueError(f"{name}.status must be one of {sorted(allowed)}")
    detail = "reason" if value["status"] == "unavailable" else "evidence"
    _text(value.get(detail), f"{name}.{detail}")


def validate_spec(spec):
    """Validate the immutable requirement record without inventing model semantics."""
    if spec.get("version") == 1:
        for key in ("target", "model_revision", "precision"):
            _text(spec.get(key), key)
        upstream = _mapping(spec.get("upstream"), "upstream")
        _text(upstream.get("repository"), "upstream.repository")
        _text(upstream.get("commit"), "upstream.commit")
        checkpoint = _mapping(spec.get("checkpoint"), "checkpoint")
        _text(checkpoint.get("id"), "checkpoint.id")
        _text(checkpoint.get("source"), "checkpoint.source")
        hardware = _mapping(spec.get("hardware"), "hardware")
        _text(hardware.get("name"), "hardware.name")
        _text(hardware.get("deployment"), "hardware.deployment")
        _mapping(spec.get("shape_profile"), "shape_profile")
    elif spec.get("version") == 2:
        target = _mapping(spec.get("target"), "target")
        for key in ("target", "hardware", "model", "model_revision", "inference_signature"):
            _text(target.get(key), "target." + key)
        _mapping(target.get("shape"), "target.shape")
        if campaign.target_key(dict(schema_version=3, **target)) != target:
            raise ValueError("Target contains checkpoint or implementation provenance")
        weights = _mapping(spec.get("initial_weights"), "initial_weights")
        for key in ("checkpoint_id", "checkpoint_digest"):
            _text(weights.get(key), "initial_weights." + key)
        variant = _mapping(spec.get("execution_variant"), "execution_variant")
        ExecutionVariant.from_dict(variant)
        reference = _mapping(spec.get("reference"), "reference")
        for key in ("repository", "commit"):
            _text(reference.get(key), "reference." + key)
        _text(spec.get("deployment"), "deployment")
        if set(spec) & {"checkpoint", "model_revision", "precision", "upstream", "hardware", "shape_profile"}:
            raise ValueError("v2 separates Target architecture, initial_weights, variant and reference")
    else:
        raise ValueError("onboarding spec version must be 1 or 2")
    objective = _mapping(spec.get("performance_objective"), "performance_objective")
    for key in ("metric", "direction", "unit"):
        _text(objective.get(key), f"performance_objective.{key}")
    requirements = _list(spec.get("correctness_requirements"),
                         "correctness_requirements", nonempty=True)
    for index, item in enumerate(requirements):
        _text(item, f"correctness_requirements[{index}]")
    protocol = _mapping(spec.get("benchmark_protocol"), "benchmark_protocol")
    _text(protocol.get("id"), "benchmark_protocol.id")
    bound = _mapping(spec.get("deployment_bound"), "deployment_bound")
    for key in ("metric", "comparator", "unit"):
        _text(bound.get(key), f"deployment_bound.{key}")
    if not isinstance(bound.get("value"), (int, float)):
        raise ValueError("deployment_bound.value must be numeric")
    budget = _mapping(spec.get("optimization_budget"), "optimization_budget")
    for key in ("candidates", "non_improving", "jobs"):
        if not isinstance(budget.get(key), int) or budget[key] <= 0:
            raise ValueError(f"optimization_budget.{key} must be a positive integer")
    assumptions = _list(spec.get("assumptions"), "assumptions")
    for index, item in enumerate(assumptions):
        _text(item, f"assumptions[{index}]")
    return spec


def _latest_attempts(directory, stages):
    result = {}
    for stage in stages:
        paths = sorted((Path(directory) / "stages" / stage).glob("attempt-*.json"))
        if paths:
            result[stage] = store.read(paths[-1])
    return result


def rebuild(directory):
    directory = Path(directory).resolve()
    metadata = store.read(directory / "onboarding.json")
    validate_spec(metadata["spec"])
    stages = _stages(metadata["spec"])
    attempts = _latest_attempts(directory, stages)
    completed = []
    current = stages[0]
    status = "ACTIVE"
    for stage in stages:
        evidence = attempts.get(stage)
        if evidence is None:
            current = stage
            break
        if evidence["status"] != "passed":
            current = stage
            status = "BLOCKED"
            break
        if metadata["spec"]["version"] == 2:
            _validate_stage(stage, evidence, metadata["spec"])
        completed.append(stage)
    else:
        current = None
        status = ("READY_FOR_OPTIMIZATION" if metadata["spec"]["version"] == 2
                  else "LEGACY_REVALIDATION_REQUIRED")
    state = {
        "version": 1,
        "target": metadata["spec"]["target"],
        "status": status,
        "completed_stages": completed,
        "current_stage": current,
        "next_action": ("create an explicit v2 spec and revalidate legacy onboarding"
                        if status == "LEGACY_REVALIDATION_REQUIRED" else
                        "begin autonomous optimization from the recorded campaign"
                        if current is None else
                        f"record {current}" if status == "ACTIVE" else f"retry {current}"),
        "attempts": {stage: len(list((directory / "stages" / stage).glob("attempt-*.json")))
                     for stage in stages},
    }
    store.write(directory / "state.json", state)
    return state


def create(directory, spec):
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(directory)
    validate_spec(spec)
    if spec["version"] != 2:
        raise ValueError("new onboarding requires v2; legacy evidence remains readable")
    store.write(directory / "onboarding.json",
                {"version": 1, "created": time.time(), "spec": spec})
    return record(directory, "requirement_freeze",
                  {"status": "passed", "spec": "onboarding.json"})


def _validate_upstream_freeze(spec, evidence):
    for key, expected in (("repository", spec["upstream"]["repository"]),
                          ("commit", spec["upstream"]["commit"]),
                          ("checkpoint", spec["checkpoint"]["id"]),
                          ("model_revision", spec["model_revision"])):
        if evidence.get(key) != expected:
            raise ValueError(f"upstream_freeze.{key} does not match onboarding spec")


def _validate_official_reference(evidence):
    checks = _mapping(evidence.get("checks"), "official_reference.checks")
    for name in OFFICIAL_REFERENCE_CHECKS:
        _check_result(checks.get(name), f"official_reference.checks.{name}",
                      unavailable=name == "upstream_official_optimized")


def build_compatibility_report(spec, inventory):
    validate_spec(spec)
    inventory = _list(inventory, "compatibility inventory", nonempty=True)
    counts = {category: 0 for category in COMPATIBILITY_CATEGORIES}
    risks = {risk: [] for risk in COMPATIBILITY_RISKS}
    components = []
    local_ops = []
    runtime_primitives = []
    for index, item in enumerate(inventory):
        item = _mapping(item, f"inventory[{index}]")
        name = _text(item.get("name"), f"inventory[{index}].name")
        category = item.get("category")
        if category not in COMPATIBILITY_CATEGORIES:
            raise ValueError(f"inventory[{index}].category is unknown: {category!r}")
        _text(item.get("evidence"), f"inventory[{index}].evidence")
        _text(item.get("action"), f"inventory[{index}].action")
        item_risks = _list(item.get("risks"), f"inventory[{index}].risks")
        unknown = sorted(set(item_risks) - set(COMPATIBILITY_RISKS))
        if unknown:
            raise ValueError(f"inventory[{index}] has unknown risks: {unknown}")
        counts[category] += 1
        for risk in item_risks:
            risks[risk].append(name)
        if category == "existing shared component":
            components.append(name)
        elif category == "Target-local op":
            local_ops.append(name)
        elif category == "genuine runtime primitive":
            runtime_primitives.append(name)
    existing_vocabulary = counts["existing runtime op"]
    return {
        "version": 1,
        "target": spec["target"]["target"] if spec["version"] == 2 else spec["target"],
        "summary": {"total": len(inventory), "by_category": counts,
                    "existing_runtime_vocabulary_count": existing_vocabulary,
                    "existing_runtime_vocabulary_fraction": existing_vocabulary / len(inventory)},
        "questions": {
            "existing_runtime_vocabulary": existing_vocabulary,
            "reusable_components": components,
            "new_target_local_ops": local_ops,
            "runtime_primitive_required": runtime_primitives,
            "dynamic_shape": risks["dynamic shape"],
            "post_freeze_allocation": risks["post-freeze allocation"],
            "host_device_sync": risks["host/device sync"],
            "graph_capture_blocker": sorted(set(risks["graph capture blocker"] + [
                item["name"] for item in inventory if item["category"] == "capture blocker"])),
            "data_dependent_control_flow": risks["data-dependent control flow"],
        },
        "inventory": inventory,
    }


def render_compatibility_markdown(report):
    counts = report["summary"]["by_category"]
    questions = report["questions"]
    lines = [f"# Compatibility report: {report['target']}", "", "## Classification", "",
             "| category | count |", "|---|---:|"]
    lines.extend(f"| {category} | {counts[category]} |" for category in COMPATIBILITY_CATEGORIES)
    lines.extend(["", "## Required questions", ""])
    labels = {
        "existing_runtime_vocabulary": "Existing runtime vocabulary coverage",
        "reusable_components": "Reusable existing components",
        "new_target_local_ops": "New Target-local ops",
        "runtime_primitive_required": "Required runtime primitives",
        "dynamic_shape": "Dynamic shapes",
        "post_freeze_allocation": "Post-freeze allocations",
        "host_device_sync": "Host/device synchronization",
        "graph_capture_blocker": "Graph-capture blockers",
        "data_dependent_control_flow": "Data-dependent control flow",
    }
    for key, label in labels.items():
        value = questions[key]
        if key == "existing_runtime_vocabulary":
            value = f"{value}/{report['summary']['total']} computation items"
        elif isinstance(value, list):
            value = ", ".join(value) if value else "none observed"
        lines.append(f"- **{label}:** {value}")
    lines.extend(["", "## Inventory", "", "| computation | classification | evidence | action | risks |",
                  "|---|---|---|---|---|"])
    for item in report["inventory"]:
        risks = ", ".join(item.get("risks", [])) or "none observed"
        lines.append(f"| {item['name']} | {item['category']} | {item['evidence']} | "
                     f"{item['action']} | {risks} |")
    return "\n".join(lines) + "\n"


def write_compatibility_report(directory, inventory):
    directory = Path(directory).resolve()
    state = rebuild(directory)
    if state["current_stage"] != "compatibility_scan":
        raise RuntimeError(f"next onboarding stage is {state['current_stage']}, not compatibility_scan")
    metadata = store.read(directory / "onboarding.json")
    report = build_compatibility_report(metadata["spec"], inventory)
    store.write(directory / "compatibility-report.json", report)
    (directory / "compatibility-report.md").write_text(render_compatibility_markdown(report))
    record(directory, "compatibility_scan", {"status": "passed", "report": report})
    return report


def _validate_target_bring_up(evidence):
    checks = _mapping(evidence.get("checks"), "target_bring_up.checks")
    for name in TARGET_BRING_UP_CHECKS:
        _check_result(checks.get(name), f"target_bring_up.checks.{name}")
    runtime = _mapping(evidence.get("runtime_modification"),
                       "target_bring_up.runtime_modification")
    if not isinstance(runtime.get("required"), bool):
        raise ValueError("target_bring_up.runtime_modification.required must be boolean")
    _text(runtime.get("reason"), "target_bring_up.runtime_modification.reason")
    if runtime["required"]:
        _text(runtime.get("unexpressible_invariant"),
              "target_bring_up.runtime_modification.unexpressible_invariant")
        runtime_checks = _mapping(runtime.get("checks"),
                                  "target_bring_up.runtime_modification.checks")
        for name in RUNTIME_MODIFICATION_CHECKS:
            _check_result(runtime_checks.get(name),
                          f"target_bring_up.runtime_modification.checks.{name}")


def _validate_ladder(evidence, expected, name, *, unavailable=()):
    entries = _list(evidence.get("ladder"), f"{name}.ladder")
    if [item.get("name") for item in entries] != list(expected):
        raise ValueError(f"{name}.ladder must use the fixed order")
    for index, item in enumerate(entries):
        _check_result(item, f"{name}.ladder[{index}]", unavailable=item["name"] in unavailable)


def _validate_floor_profile(evidence):
    call_sites = _list(evidence.get("call_sites"), "floor_profile.call_sites", nonempty=True)
    for index, item in enumerate(call_sites):
        item = _mapping(item, f"floor_profile.call_sites[{index}]")
        for key in ("name", "geometry"):
            _text(item.get(key), f"floor_profile.call_sites[{index}].{key}")
        for key in ("minimal_bytes", "flops", "recoverable_latency_ms"):
            if not isinstance(item.get(key), (int, float)) or item[key] < 0:
                raise ValueError(f"floor_profile.call_sites[{index}].{key} must be non-negative")
        ceiling = _mapping(item.get("measured_ceiling"),
                           f"floor_profile.call_sites[{index}].measured_ceiling")
        if ceiling.get("geometry") != item["geometry"]:
            raise ValueError("measured ceiling geometry must match its call site")
        for key in ("metric", "unit", "evidence"):
            _text(ceiling.get(key), f"floor_profile.call_sites[{index}].measured_ceiling.{key}")
        if not isinstance(ceiling.get("value"), (int, float)) or ceiling["value"] <= 0:
            raise ValueError("measured ceiling value must be positive")
    _text(evidence.get("profile"), "floor_profile.profile")


def _validate_legacy_campaign(evidence, spec):
    for key in ("directory", "objective", "protocol", "fixture", "state"):
        _text(evidence.get(key), f"campaign_creation.{key}")
    if evidence["objective"] != spec["performance_objective"]["metric"]:
        raise ValueError("campaign_creation.objective does not match onboarding spec")
    if evidence["protocol"] != spec["benchmark_protocol"]["id"]:
        raise ValueError("campaign_creation.protocol does not match onboarding spec")
    if evidence["state"] != "BASELINED":
        raise ValueError("a new Target campaign must start BASELINED")



def _campaign_key(spec):
    return campaign.campaign_key(
        dict(schema_version=3, **spec["target"], execution_variant=spec["execution_variant"]),
        objective=spec["performance_objective"]["metric"], protocol=spec["benchmark_protocol"]["id"])


def _initial_context(spec, context):
    measurement.context(context)
    for key in ("checkpoint_id", "checkpoint_digest"):
        if context["weights"][key] != spec["initial_weights"][key]:
            raise ValueError("Campaign checkpoint differs from initial_weights")
    for key in ("repository", "commit"):
        if context["reference_provenance"].get(key) != spec["reference"][key]:
            raise ValueError("anchor reference provenance differs from onboarding reference")


def _validate_handoff(evidence, spec, *, published=False, current=False):
    root = Path(evidence["repository"]).resolve()
    key = _campaign_key(spec)
    location = CampaignRegistry(root).find(key)
    if location is None or location != Path(evidence["directory"]).resolve():
        raise ValueError("handoff requires the actual canonical Campaign")
    metadata = store.read(location / "campaign.json")
    if metadata["id"] != evidence["campaign_id"]:
        raise ValueError("handoff lineage differs from Campaign")
    if current:
        state = store.read(location / "state.json")
        if (state["reanchor_required"] or state["current_stage"] not in ("candidate_selection", "publication")
                or evidence["segment"] != state["current_measurement_segment"]):
            raise ValueError("new handoff requires the current validated anchor and no active experiment")
    baseline = campaign._records(location)[0][1]["measurement"]["evidence"]
    segments = transition.segments(location, baseline)
    number = evidence["segment"]
    if type(number) is not int or not 0 <= number < len(segments):
        raise ValueError("handoff segment is not a registered anchor")
    segment = segments[number]
    _initial_context(spec, segment["measurement"]["measurement_context"])
    measurement.latency(segment["measurement"], expected_identity=segment["measurement"]["identity"],
                        expected_context=segment["measurement"]["measurement_context"],
                        protocol=spec["benchmark_protocol"]["id"],
                        objective=spec["performance_objective"]["metric"])
    measurement.correctness(segment["measurement"]["correctness"], segment["measurement"]["identity"],
                            segment["measurement"]["measurement_context"])
    if published:
        if current and state["publication_required"]:
            raise ValueError("Campaign publication is pending; publish before recording readiness")
        from lab.results import schema
        from lab.optimize import trace

        destination = root / "results/targets" / key_digest(key)
        summary = schema.check_views(destination)
        value = store.read(destination / "trace.json")
        current_trace = trace.normalize(location)
        if current and value != schema.compact_trace(current_trace):
            raise ValueError("current Campaign publication is pending; published history is stale")
        if summary["campaign_key"] != key or summary["lineage_id"] != metadata["id"]:
            raise ValueError("published baseline belongs to another Campaign")
        if value["segments"][evidence["segment"]] != current_trace["segments"][evidence["segment"]]:
            raise ValueError("published handoff segment differs from the validated anchor")
        index = store.read(root / "results/index.json")
        expected = dict(campaign_key=key, lineage_id=metadata["id"],
                        summary=(destination / "summary.json").relative_to(root / "results").as_posix())
        matches = [entry for entry in index["campaigns"]
                   if entry["summary"] == expected["summary"]]
        if index["schema_version"] != 1 or matches != [expected]:
            raise ValueError("published Campaign discovery entry is missing or inconsistent")


def _validate_stage(stage, evidence, spec, *, new=False):
    if evidence.get("status") not in ("passed", "failed", "blocked"):
        raise ValueError(f"{stage}.status must be passed, failed or blocked")
    if evidence["status"] != "passed":
        _text(evidence.get("reason"), f"{stage}.reason")
        return
    if stage == "requirement_freeze":
        if evidence.get("spec") != "onboarding.json":
            raise ValueError("requirement_freeze must cite onboarding.json")
    elif stage == "upstream_freeze":
        _validate_upstream_freeze(spec, evidence)
    elif stage == "reference_freeze":
        for key in ("repository", "commit"):
            if evidence.get(key) != spec["reference"][key]:
                raise ValueError("reference provenance does not match onboarding spec")
    elif stage in ("model_contract", "weights_compatibility"):
        contract = _mapping(evidence.get("contract"), stage + ".contract")
        if inference_signature(**contract) != spec["target"]["inference_signature"]:
            raise ValueError("inference signature mismatch; resolve a compatible Target/model revision")
        if stage == "weights_compatibility":
            for key in ("checkpoint_id", "checkpoint_digest"):
                if evidence["weights"].get(key) != spec["initial_weights"][key]:
                    raise ValueError("compatibility checked different initial weights")
    elif stage == "official_reference":
        _validate_official_reference(evidence)
    elif stage == "compatibility_scan":
        report = _mapping(evidence.get("report"), "compatibility_scan.report")
        rebuilt = build_compatibility_report(spec, report.get("inventory"))
        if report != rebuilt:
            raise ValueError("compatibility report contains non-derived fields")
    elif stage == "target_bring_up":
        _validate_target_bring_up(evidence)
    elif stage == "correctness_ladder":
        _validate_ladder(evidence, CORRECTNESS_LADDER, "correctness_ladder")
    elif stage == "baseline_ladder":
        _validate_ladder(evidence, BASELINE_LADDER, "baseline_ladder",
                         unavailable={"upstream official optimized/compile"})
    elif stage == "floor_profile":
        _validate_floor_profile(evidence)
    elif stage in ("campaign_creation", "publication"):
        if spec["version"] == 1:
            _validate_legacy_campaign(evidence, spec)
        else:
            _validate_handoff(evidence, spec, published=stage == "publication", current=new)


def record(directory, stage, evidence):
    directory = Path(directory).resolve()
    metadata = store.read(directory / "onboarding.json")
    if stage not in _stages(metadata["spec"]):
        raise ValueError(f"unknown onboarding stage {stage!r}")
    state = rebuild(directory)
    if state["current_stage"] != stage:
        raise RuntimeError(f"next onboarding stage is {state['current_stage']}, not {stage}")
    _validate_stage(stage, evidence, metadata["spec"], new=True)
    stage_dir = directory / "stages" / stage
    attempt = len(list(stage_dir.glob("attempt-*.json"))) + 1
    value = dict(evidence, stage=stage, attempt=attempt, timestamp=time.time())
    store.write(stage_dir / f"attempt-{attempt:03d}.json", value)
    return rebuild(directory)


def validate(directory):
    directory = Path(directory).resolve()
    metadata = store.read(directory / "onboarding.json")
    spec = validate_spec(metadata["spec"])
    for stage, evidence in _latest_attempts(directory, _stages(spec)).items():
        _validate_stage(stage, evidence, spec)
    return rebuild(directory)



def handoff(directory, root, baseline, *, inputs=(), transition_request=None):
    """Create/restore the actual lineage and publish before readiness; never rerun a failed job."""
    directory, root = Path(directory).resolve(), Path(root).resolve()
    with (directory / "handoff.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = validate(directory)
        if state["status"] == "READY_FOR_OPTIMIZATION":
            return state
        if state["current_stage"] not in ("campaign_creation", "publication"):
            raise RuntimeError("complete onboarding evidence before Campaign handoff")
        spec = store.read(directory / "onboarding.json")["spec"]
        if spec["version"] != 2:
            raise ValueError("handoff requires an explicit v2 onboarding spec")
        key = _campaign_key(spec)
        from lab.optimize import policy as acceptance

        if spec["optimization_budget"] != acceptance.for_target(spec["target"]["target"])["budget"]:
            raise ValueError("onboarding budget differs from the registered Target budget")
        if spec["performance_objective"]["direction"] != "minimize" or spec["performance_objective"]["unit"] != "ms":
            raise ValueError("Campaign latency objective requires minimize and ms")
        if campaign.campaign_key(baseline["identity"], objective=key["objective"],
                                 protocol=key["benchmark_protocol"]) != key:
            raise ValueError("baseline workload differs from onboarding Target/variant")
        _initial_context(spec, baseline["measurement_context"])
        location = CampaignRegistry(root, require_portable_source=True).open_or_seed(
            key, baseline=baseline, inputs=inputs)
        current = campaign.configure_publication(location, root)
        expected = measurement.context(baseline["measurement_context"])
        active = measurement.context(current["measurement_context"])
        reference_changed = any(active.reference_provenance.get(key) != spec["reference"][key]
                                for key in ("repository", "commit"))
        if current["reanchor_required"] or active.segment_key != expected.segment_key or reference_changed:
            if transition_request is None:
                raise ValueError("existing Campaign needs context validation and re-anchor; supply a transition request")
            requested = measurement.context(transition_request["measurement_context"])
            if requested.segment_key != expected.segment_key:
                raise ValueError("transition request differs from onboarding checkpoint/fixture/environment")
            _initial_context(spec, transition_request["measurement_context"])
            if current["pending_transition"]:
                raise RuntimeError("resume or reconcile the pending Campaign transition before handoff")
            current = campaign.transition_context(root, location, transition_request)
        if current["current_stage"] not in ("candidate_selection", "publication"):
            raise RuntimeError("finish the active Campaign work before onboarding handoff")
        receipt = dict(status="passed", repository=str(root), directory=str(location),
                       campaign_id=current["campaign_id"], segment=current["current_measurement_segment"])
        if state["current_stage"] == "campaign_creation":
            record(directory, "campaign_creation", receipt)
        from lab.results.publish import publish

        # Republishing is recoverable and never executes an experiment.
        publish(root, location)
        campaign.rebuild(location)
        return record(directory, "publication", receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "record", "compatibility-report",
                                            "status", "validate", "handoff"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("input", nargs="?", type=Path,
                        help="spec, stage evidence, or compatibility inventory JSON")
    parser.add_argument("--stage", choices=tuple(dict.fromkeys((*STAGES, *LEGACY_STAGES))))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-input", action="append", default=[])
    parser.add_argument("--transition-request", type=Path)
    args = parser.parse_args(argv)
    if args.command == "init":
        if args.input is None:
            parser.error("init requires a spec JSON")
        result = create(args.directory, store.read(args.input))
    elif args.command == "handoff":
        if args.input is None:
            parser.error("handoff requires normalized baseline evidence")
        result = handoff(args.directory, args.root, store.read(args.input), inputs=args.source_input,
                         transition_request=(store.read(args.transition_request) if args.transition_request else None))
    elif args.command == "record":
        if args.stage is None or args.input is None:
            parser.error("record requires --stage and an evidence JSON")
        result = record(args.directory, args.stage, store.read(args.input))
    elif args.command == "compatibility-report":
        if args.input is None:
            parser.error("compatibility-report requires an inventory JSON")
        result = write_compatibility_report(args.directory, store.read(args.input))
    elif args.command == "status":
        result = rebuild(args.directory)
    else:
        result = validate(args.directory)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
