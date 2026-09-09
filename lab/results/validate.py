"""Published-fact integrity; this does not recreate or certify raw GPU evidence."""
import math

from lab.optimize import campaign, measurement
from lab.optimize.schema import validate_applicability


def _equal(actual, expected, field):
    if actual != expected:
        raise ValueError(f"published trace {field} mismatch")


def _number(value, field, *, optional=False):
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"published trace {field} must be finite positive latency")


def _arithmetic(actual, expected, field):
    if expected is None:
        _equal(actual, None, field)
    elif (isinstance(actual, bool) or not isinstance(actual, (int, float))
          or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10)):
        raise ValueError(f"published trace {field} arithmetic mismatch")


def _row_identity(row, variant):
    return dict(schema_version=3, **row["target_key"], execution_variant=variant,
                engine_revision=row["engine_revision"], plan=row["plan"])


def _segment(segment, *, number, boundary, total, incumbent, identity, protocol):
    _equal(segment["id"], number, "segment ID")
    if type(segment["id"]) is not int or (number and segment["before_iteration"] == 0):
        raise ValueError("published trace invalid segment ID/boundary")
    if type(segment["before_iteration"]) is not int or not boundary <= segment["before_iteration"] <= total:
        raise ValueError("published trace segment boundary is not monotonic")
    if number == 0:
        _equal(segment["before_iteration"], 0, "baseline boundary")
    _equal(segment["incumbent"], f"iter-{incumbent:03d}", "anchor incumbent")
    measurement.identity(segment["identity"], identity)
    _number(segment["anchor_ms"], "anchor")
    context = measurement.context(segment["measurement_context"])
    descriptor = dict(id=number, benchmark_protocol=protocol, **context.environment)
    return context, descriptor


def _row(row, *, number, target, variant, incumbent, context, descriptor, best, anchor, baseline):
    _equal(row["iteration"], number, "iteration ID")
    if type(row["iteration"]) is not int:
        raise ValueError("published trace iteration ID must be integer")
    _equal(row["target_key"], target, "TargetKey")
    _equal(row["parent_incumbent"], None if number == 0 else f"iter-{incumbent:03d}", "parent incumbent")
    identity = _row_identity(row, variant)
    measurement.identity(identity, identity)
    observed = measurement.context(row["measurement_context"])
    _equal(observed.segment_key, context.segment_key, "context/segment")
    _equal(row["context_id"], observed.context_id, "context ID")
    _equal(row["measurement_segment"], descriptor, "segment environment/protocol")
    _equal(row["reanchor"], False, "iteration cannot be a context re-anchor")
    verdict = row["verdict"]
    if verdict not in campaign.TERMINAL:
        raise ValueError("published trace contains non-terminal verdict")
    candidate, parent = row["candidate_latency_ms"], row["parent_incumbent_latency_ms"]
    _number(candidate, "candidate", optional=verdict not in ("accepted", "no_benefit"))
    _number(parent, "parent", optional=number == 0 or verdict not in ("accepted", "no_benefit"))
    if verdict in ("accepted", "no_benefit"):
        _equal(row["measurement_validity"], "valid", "performance validity")
        _equal(row["correctness"]["status"], "pass", "correctness status")
    if number == 0:
        _equal(verdict, "accepted", "baseline verdict")
        _equal(parent, None, "baseline parent")
        _arithmetic(candidate, baseline, "baseline candidate")
        portable = True
    else:
        portable = validate_applicability(row) != "checkpoint_specific"
        if verdict == "accepted":
            _equal(row["qualification"]["status"], "pass", "qualification status")
            _equal(row["qualification"]["gate_verdict"], "pass", "qualification verdict")
        elif verdict == "correctness_failed":
            _equal(row["correctness"]["status"], "failed", "failed correctness")
        elif verdict == "invalid" and row["measurement_validity"] == "valid":
            raise ValueError("invalid candidate has valid measurement")
    promoted = verdict == "accepted" and portable
    expected_promotion = ("promoted" if portable else "context_only") if verdict == "accepted" else "not_promoted"
    _equal(row["promotion"], expected_promotion, "promotion")
    if promoted:
        best, incumbent = candidate, number
    _arithmetic(row["current_incumbent_latency_ms"], best, "current incumbent")
    _arithmetic(row["campaign_baseline_latency_ms"], baseline, "campaign baseline")
    _arithmetic(row["segment_baseline_latency_ms"], anchor, "segment anchor")
    delta_parent = None if candidate is None or parent is None else (candidate / parent - 1) * 100
    delta_anchor = 0. if number == 0 else None if candidate is None else (candidate / anchor - 1) * 100
    _arithmetic(row["delta_vs_parent_pct"], delta_parent, "delta parent")
    _arithmetic(row["delta_vs_baseline_pct"], delta_anchor, "delta context anchor")
    return best, incumbent


def trace(value):
    """Verify identity, context, lineage and arithmetic from retained terminal facts."""
    _equal(value["schema_version"], 1, "schema version")
    rows, segments, metadata = value["iterations"], value["segments"], value["metadata"]
    if not rows or not segments:
        raise ValueError("published trace requires a baseline and segment")
    target, variant = rows[0]["target_key"], metadata["execution_variant"]
    for field in ("hardware", "model", "model_revision", "inference_signature", "shape"):
        _equal(metadata[field], target[field], f"metadata {field}")
    _equal(metadata["precision"], variant["quantization"]["mode"], "ExecutionVariant precision")
    _equal(metadata["shape_profile"], "-".join(f"{k}{v}" for k, v in sorted(target["shape"].items())), "shape profile")
    _equal(metadata["fixture"], segments[0]["measurement_context"]["fixture"]["id"], "initial fixture")
    incumbent, cursor, boundary = 0, 0, 0
    baseline = segments[0]["anchor_ms"]
    best = anchor = baseline
    context = descriptor = None
    for number in range(len(rows) + 1):
        while cursor < len(segments) and segments[cursor]["before_iteration"] == number:
            segment = segments[cursor]
            context, descriptor = _segment(
                segment, number=cursor, boundary=boundary, total=len(rows), incumbent=incumbent,
                identity=_row_identity(rows[incumbent], variant), protocol=metadata["protocol"])
            best = anchor = segment["anchor_ms"]
            boundary, cursor = number, cursor + 1
        if number == len(rows):
            break
        if context is None:
            raise ValueError("published trace has no initial segment")
        best, incumbent = _row(
            rows[number], number=number, target=target, variant=variant, incumbent=incumbent,
            context=context, descriptor=descriptor, best=best, anchor=anchor, baseline=baseline)
    if cursor != len(segments):
        raise ValueError("published trace segment boundary is not monotonic or is outside history")
    key = campaign.campaign_key(_row_identity(rows[0], variant),
                                objective=metadata["objective"], protocol=metadata["protocol"])
    _equal(target, key["target"], "canonical TargetKey")
    return key
