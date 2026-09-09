"""Compact continuation facts and committed portable source; never raw experiment replay."""
from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile

from lab.optimize import campaign, store, trace, transition
from lab.optimize.schema import relative_path
from . import schema


def _pick(value, names):
    return {name: deepcopy(value[name]) for name in names.split() if name in value}


def _correctness(value):
    return _pick(value, "status checks identity measurement_context")


def _measurement(value):
    result = _pick(value, "validity identity measurement_context measurement_segment protocol instrumented "
                         "objective candidate_ms parent_incumbent_ms segment_anchor_ms reanchor")
    if "legacy_history" in value:
        result["legacy_history"] = {
            category: [dict(
                _pick(item, "source line evidence_level"),
                record=_pick(item["record"], "id thesis status hypothesis conclusion reason conditions reopen_when"))
                for item in items]
            for category, items in value["legacy_history"].items()}
    if "correctness" in value:
        result["correctness"] = _correctness(value["correctness"])
    if "aba" in value:
        aba = value["aba"]
        result["aba"] = dict(
            config=_pick(aba["config"], "reps warmup soak_s p99_min_reps"),
            legs=[dict(identity=leg["identity"], measurement_context=leg["measurement_context"],
                       metrics={name: _pick(stats, "min median p99 n") for name, stats in leg["metrics"].items()})
                  for leg in aba["legs"]])
    return result


def _history(directory):
    metadata = _pick(store.read(directory / "campaign.json"),
                     "version id created target objective protocol fixture budget execution_variant")
    original = store.read(directory / "campaign.json")
    if "fork" in original:
        metadata["fork"] = _pick(original["fork"], "parent_campaign parent_iteration reason")
    records = []
    for _, record in campaign._records(directory):
        row = _pick(record, "iteration candidate_id parent_incumbent timestamp hypothesis change cost verdict "
                            "kind campaign_identity")
        row["correctness"] = _correctness(record["correctness"])
        row["qualification"] = _pick(record["qualification"], "status gate_verdict")
        row["measurement"] = _measurement(record["measurement"])
        if row["iteration"] == 0:
            row["measurement"]["evidence"] = _measurement(record["measurement"]["evidence"])
        else:
            row["spec"] = _pick(record["spec"], "protocol fixture measurement_context measurement_segment "
                                               "applicability artifact_recipe retune_recipe")
        row["diagnostics"] = {}
        records.append(row)
    transitions = []
    for _, record in transition.records(directory):
        if record["status"] not in ("activated", "aborted"):
            raise ValueError("cannot snapshot a pending context transition")
        item = _pick(record, "status cost")
        if record["status"] == "activated":
            item["segment"] = _pick(record["segment"], "id before_iteration incumbent")
            item["segment"]["measurement"] = _measurement(record["segment"]["measurement"])
        transitions.append(item)
    return dict(metadata=metadata, records=records, transitions=transitions,
                hypotheses=dict(unresolved=campaign.rebuild(directory)["highest_value_unresolved_hypotheses"]))


def _write_history(directory, history):
    store.write(directory / "hypotheses.json", history["hypotheses"])
    for number, record in enumerate(history["records"]):
        if record.get("verdict") not in campaign.TERMINAL:
            raise ValueError("snapshot contains an unfinished iteration")
        store.write(directory / "runs" / f"iter-{number:03d}-published" / "evidence.json", record)
    for number, record in enumerate(history["transitions"]):
        if record["status"] not in ("activated", "aborted"):
            raise ValueError("snapshot contains an unfinished transition")
        store.write(directory / "transitions" / f"transition-{number:03d}" / "evidence.json", record)
    store.write(directory / "campaign.json", history["metadata"])


def reconstruct(snapshot):
    """Revalidate retained normalized receipts with the existing ledger rules."""
    if snapshot["schema_version"] != 1:
        raise ValueError("unsupported resume snapshot version")
    with tempfile.TemporaryDirectory(prefix="flash-vla-resume-") as temporary:
        directory = Path(temporary)
        _write_history(directory, snapshot["history"])
        state = campaign.rebuild(directory)
        if state["reanchor_required"]:
            raise ValueError("snapshot history ends without a validated anchor")
        value = schema.compact_trace(trace.normalize(directory))
        if "fork" in snapshot["history"]["metadata"]:
            value["fork"] = snapshot["history"]["metadata"]["fork"]
    summary, _ = schema.summaries(value)
    incumbent = int(state["portable_incumbent"].split("-")[1])
    row = value["iterations"][incumbent]
    source = snapshot["portable_source"]
    if source["revision"] != row["engine_revision"] or not source["inputs"]:
        raise ValueError("snapshot portable source does not match incumbent")
    for name in source["inputs"]:
        relative_path(name)
    derived = dict(
        schema_version=1, campaign_key=summary["campaign_key"], lineage_id=summary["lineage_id"],
        next_iteration=state["iterations"],
        portable_incumbent=dict(iteration=incumbent, engine_revision=row["engine_revision"], plan=row["plan"]),
        active_context=state["active_context"], context_state=state["contexts"],
        last_segment=state["current_measurement_segment"],
        open_hypotheses=state["highest_value_unresolved_hypotheses"],
        failed_hypotheses=state["failed_hypotheses"],
        portable_source=source, history=snapshot["history"])
    if "fork" in summary:
        derived["fork"] = summary["fork"]
    return derived, value


def check(snapshot, value):
    expected, historical_trace = reconstruct(snapshot)
    if expected != snapshot:
        raise ValueError("resume snapshot derived state is stale")
    if historical_trace != value:
        raise ValueError("resume snapshot differs from published trace")
    return expected


def build(root, directory, value):
    """Export small validated facts; prove only declared incumbent inputs are committed."""
    state = campaign.rebuild(directory)
    record = campaign.incumbent_record(directory, state)
    if "source" not in record:
        raise ValueError("published resume requires declared portable source inputs")
    source = record["source"]
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", source["revision"] + "^{commit}"], cwd=root, text=True).strip()
    if revision != record["change"]["engine_revision"]:
        raise ValueError("portable source must name its committed engine revision")
    for name in source["inputs"]:
        relative_path(name)
        committed = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root)
        if committed != (Path(source["root"]) / name).read_bytes():
            raise ValueError("portable source differs from committed engine; commit and requalify")
    snapshot, rebuilt = reconstruct(dict(
        schema_version=1, history=_history(directory),
        portable_source=dict(revision=revision, inputs=source["inputs"])))
    if (snapshot["failed_hypotheses"] != state["failed_hypotheses"]
            or snapshot["open_hypotheses"] != state["highest_value_unresolved_hypotheses"]):
        raise ValueError("compact snapshot lost campaign hypotheses")
    if rebuilt != schema.compact_trace(value):
        raise ValueError("compact snapshot lost published trace facts")
    return snapshot


def seed(root, directory, snapshot, value):
    """Import historical facts, extract committed source, and require a new anchor."""
    check(snapshot, value)
    if snapshot["lineage_id"] != directory.name:
        raise ValueError("snapshot lineage does not match canonical local directory")
    revision = snapshot["portable_source"]["revision"]
    resolved = subprocess.check_output(
        ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=root, text=True).strip()
    if resolved != revision:
        raise ValueError("snapshot engine revision is not an exact available commit")
    # Validate and extract before reserving the canonical directory.
    source_bytes = {name: subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root)
                    for name in snapshot["portable_source"]["inputs"]}
    directory.mkdir(parents=True, exist_ok=False)
    history = deepcopy(snapshot["history"])
    history["metadata"]["published_import"] = dict(
        next_iteration=snapshot["next_iteration"], last_segment=snapshot["last_segment"])
    incumbent = snapshot["portable_incumbent"]["iteration"]
    inputs = directory / "runs" / f"iter-{incumbent:03d}-published" / "inputs"
    for name, data in source_bytes.items():
        target = inputs / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    history["records"][incumbent]["source"] = dict(
        snapshot["portable_source"], root=str(inputs),
        coverage="published committed portable inputs")
    # campaign.json becomes discoverable only after every receipt and source input.
    _write_history(directory, history)
    return campaign.rebuild(directory)
