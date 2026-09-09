"""Context-local summaries; cross-context minima are never compared."""
from flash_vla.runtime.identity import MeasurementContext
from lab.optimize import store
from . import validate


def summaries(value):
    """Summarize the latest validated segment for each activated context."""
    validate.trace(value)
    contexts = {}
    for segment in value["segments"]:
        context = segment["measurement_context"]
        context_id = MeasurementContext.from_dict(context).context_id
        incumbent = int(segment["incumbent"].split("-")[1])
        identity = segment["identity"]
        best = segment["anchor_ms"]
        for row in value["iterations"]:
            if row["measurement_segment"]["id"] != segment["id"]:
                continue
            if row["verdict"] == "accepted" and row["promotion"] == "promoted":
                best, incumbent = row["candidate_latency_ms"], row["iteration"]
                identity = dict(identity, engine_revision=row["engine_revision"], plan=row["plan"])
        contexts[context_id] = dict(
            schema_version=1, context_id=context_id, weights=context["weights"],
            fixture=context["fixture"], measurement_context=context,
            latest_segment=segment["id"], segment_anchor_ms=segment["anchor_ms"],
            best_validated_ms=best,
            validated_incumbent=dict(iteration=incumbent,
                                     engine_revision=identity["engine_revision"], plan=identity["plan"]),
            last_validated_iteration=incumbent)
    active_context = MeasurementContext.from_dict(value["segments"][-1]["measurement_context"]).context_id
    representative = contexts[active_context]
    key = dict(target=value["iterations"][0]["target_key"],
               execution_variant=value["metadata"]["execution_variant"],
               objective=value["metadata"]["objective"], benchmark_protocol=value["metadata"]["protocol"])
    summary = dict(
        schema_version=1, campaign_key=key, lineage_id=value["campaign_id"],
        representative_context=active_context,
        validated_context_count=len(contexts), latest_iteration=value["iterations"][-1]["iteration"],
        current_performance=dict(
            anchor_ms=representative["segment_anchor_ms"],
            best_ms=representative["best_validated_ms"],
            speedup=representative["segment_anchor_ms"] / representative["best_validated_ms"],
            best_iteration=representative["validated_incumbent"]["iteration"]))
    if "fork" in value:
        summary["fork"] = value["fork"]
    return summary, contexts


def compact_trace(value):
    """Publish small facts; raw profiler output, logs and command receipts stay local."""
    rows = []
    for row in value["iterations"]:
        row = dict(row)
        row.pop("diagnostic_artifacts")
        row["correctness"] = {key: row["correctness"][key]
                              for key in ("status", "checks") if key in row["correctness"]}
        row["qualification"] = {key: row["qualification"][key]
                                for key in ("status", "gate_verdict") if key in row["qualification"]}
        rows.append(row)
    return dict(value, iterations=rows)


def check_views(directory):
    """Refuse incomplete/mixed publication views before indexing their performance."""
    value = store.read(directory / "trace.json")
    summary, contexts = summaries(value)
    if store.read(directory / "summary.json") != summary:
        raise ValueError("published summary is stale relative to trace; republish this lineage")
    for context_id, expected in contexts.items():
        if store.read(directory / "contexts" / context_id / "summary.json") != expected:
            raise ValueError("published context summary is stale relative to trace; republish this lineage")
    for name in ("progress.svg", "README.md"):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    from . import resume
    resume.check(store.read(directory / "resume.json"), value)
    return summary
