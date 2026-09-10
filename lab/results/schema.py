"""Context-local summaries; cross-context minima are never compared."""
from flash_vla.runtime.identity import MeasurementContext


def summaries(value):
    """Summarize the latest validated segment for each activated context."""
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
