"""Validation at measurement boundaries; no tensor or source hashing."""
import math

from flash_vla.runtime.identity import Identity, MeasurementContext

ENVIRONMENT_FIELDS = (
    "gpu_sku", "driver", "cuda_runtime", "pytorch", "tilelang",
    "clock_policy", "power_policy", "capture_regime",
)


def context(value):
    """Require complete immutable asset and measurement-environment provenance."""
    for group, keys in (
        ("weights", ("checkpoint_id", "checkpoint_digest")),
        ("fixture", ("id", "digest")),
        ("environment", ENVIRONMENT_FIELDS),
    ):
        missing = [key for key in keys if value.get(group, {}).get(key) in (None, "")]
        if missing:
            raise ValueError(f"measurement context missing {group}: {missing}")
    for key in ("hostname", "slurm_job_id", "timestamp"):
        if value.get(key) in (None, ""):
            raise ValueError(f"measurement context missing {key}")
    return MeasurementContext.from_dict(value)


def identity(value, expected):
    actual, parent = Identity.from_dict(value), Identity.from_dict(expected)
    if not actual.same_workload(parent):
        raise ValueError("inference signature / Target mismatch; resolve a compatible Target/model revision")
    if actual.plan != parent.plan or actual.engine_revision != parent.engine_revision:
        raise ValueError("measurement implementation does not match the recorded incumbent/candidate")
    if not actual.engine_revision:
        raise ValueError("measurement needs a resolved engine revision")


def compatibility(report, expected_identity, expected_context):
    """A structural check binds the Target ABI and assets, not candidate kernels."""
    if report.get("status") != "pass":
        raise ValueError("compatibility did not pass")
    if not Identity.from_dict(report["identity"]).same_workload(Identity.from_dict(expected_identity)):
        raise ValueError("inference signature mismatch; resolve a compatible Target/model revision")
    actual, expected = context(report["measurement_context"]), context(expected_context)
    if actual.context_id != expected.context_id:
        raise ValueError("compatibility checked different checkpoint/fixture assets")


def correctness(report, expected_identity, expected_context):
    """Bind correctness to the actual implementation and checkpoint/fixture."""
    if report.get("status") != "pass":
        raise ValueError("correctness did not pass")
    identity(report["identity"], expected_identity)
    observed = report["measurement_context"]
    expected = context(expected_context)
    if context(observed).segment_key != expected.segment_key:
        raise ValueError("correctness context/environment differs from requested segment")
    for name, keys in (("weights", ("checkpoint_id", "checkpoint_digest")),
                       ("fixture", ("id", "digest"))):
        if {key: observed[name][key] for key in keys} != {
                key: getattr(expected, name)[key] for key in keys}:
            raise ValueError(f"correctness {name} differs from the requested context")


def latency(report, *, expected_identity, expected_context, protocol, objective):
    """Validate normalized anchor evidence without inventing absent facts."""
    identity(report["identity"], expected_identity)
    observed, expected = context(report["measurement_context"]), context(expected_context)
    if observed.segment_key != expected.segment_key:
        raise ValueError("measurement context changed; require a new segment and re-anchor")
    if report.get("protocol") != protocol:
        raise ValueError("measurement protocol differs from campaign")
    if report.get("validity") != "valid" or report.get("instrumented") is not False:
        raise ValueError("anchor requires valid uninstrumented measurement")
    metric = report["objective"]
    if metric.get("name") != objective or metric.get("unit") != "ms":
        raise ValueError("measurement objective differs from campaign")
    value = float(metric["value"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("latency must be finite and positive")
    return value


def comparison(report, *, expected_identity, parent_identity, expected_context, protocol, objective):
    """Derive parent/candidate scalars from same-context A/B/A leg evidence."""
    from lab.optimize import policy as acceptance

    policy = acceptance.for_target(expected_identity["target"])["latency"]
    aba = report["aba"]
    if len(aba["legs"]) != 3:
        raise ValueError("comparison requires three A/B/A legs")
    if any(aba["config"].get(key) != policy[key] for key in ("warmup", "reps")):
        raise ValueError("A/B/A repetitions/warmup differ from benchmark protocol")
    metric = {"e2e_chunk_latency_ms": "chunk_latency", "device_latency_ms": "device_latency"}[objective]
    values = []
    for leg, expected in zip(aba["legs"], (parent_identity, expected_identity, parent_identity)):
        normalized = dict(identity=leg["identity"], measurement_context=leg["measurement_context"],
                          validity=report["validity"], instrumented=report["instrumented"],
                          protocol=report["protocol"],
                          objective=dict(name=objective, unit="ms", value=leg["metrics"][metric]["min"]))
        values.append(latency(normalized, expected_identity=expected,
                              expected_context=expected_context, protocol=protocol, objective=objective))
    if abs(values[0] - values[2]) > policy["control_spread_max_ms"]:
        raise ValueError("A/B/A control spread exceeds acceptance policy")
    if report["parent_incumbent_ms"] != values[0] or report["candidate_ms"] != values[1]:
        raise ValueError("parent/candidate latency must come from the validated A/B/A legs")
