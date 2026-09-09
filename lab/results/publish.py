"""Publish a terminal ledger using the canonical trace and Matplotlib renderer."""
import fcntl
from pathlib import Path
import tempfile
import os

from lab.optimize import campaign, store, trace, transition
from lab.optimize.registry import key_digest

from . import index, render, resume, schema


def publish(root, directory):
    """Publish compact facts and views after validating the local authoritative ledger."""
    root, directory = Path(root).resolve(), Path(directory).resolve()
    with (directory / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = campaign.rebuild(directory)
        metadata = store.read(directory / "campaign.json")
        if metadata.get("publication_root", str(root)) != str(root):
            raise ValueError("publication repository differs from the configured local Campaign")
        if "execution_variant" not in metadata:
            raise ValueError("published continuation requires explicitly migrated Identity v3")
        if any(record.get("verdict") is None for _, record in campaign._records(directory)):
            raise ValueError("finalize the active iteration before publishing")
        if transition.pending(directory) or state["reanchor_required"]:
            raise ValueError("finish context validation and re-anchor before publishing")
        value = trace.normalize(directory)
        if "fork" in metadata:
            value["fork"] = {key: metadata["fork"][key]
                              for key in ("parent_campaign", "parent_iteration", "reason")}
        summary, _ = schema.summaries(value)
        snapshot = resume.build(root, directory, value)
        relative = Path("targets") / key_digest(summary["campaign_key"])
        if "fork" in metadata:
            relative /= Path("forks") / metadata["id"]
        results = root / "results"
        scratch = root / "artifacts/results"
        scratch.mkdir(parents=True, exist_ok=True)
        with (scratch / "publication.lock").open("a") as publication:
            fcntl.flock(publication, fcntl.LOCK_EX)
            destination = results / relative
            trace_path = destination / "trace.json"
            if trace_path.exists():
                prior_trace = store.read(trace_path)
                previous, _ = schema.summaries(prior_trace)
                if (previous["campaign_key"] != summary["campaign_key"]
                        or previous["lineage_id"] != summary["lineage_id"]
                        or previous.get("fork") != summary.get("fork")):
                    raise ValueError("published CampaignKey already belongs to another lineage")
                current = schema.compact_trace(value)
                if (current["iterations"][:len(prior_trace["iterations"])] != prior_trace["iterations"]
                        or current["segments"][:len(prior_trace["segments"])] != prior_trace["segments"]):
                    raise ValueError("publication cannot rewrite or truncate validated history")
            elif destination.exists() and any(
                    path.is_file() and not path.is_relative_to(destination / "forks")
                    for path in destination.rglob("*")):
                raise ValueError("published files have no trace owner; reconcile before publishing")
            with tempfile.TemporaryDirectory(dir=scratch, prefix="publish-") as temporary:
                staged = Path(temporary)
                store.write(staged / "trace.json", schema.compact_trace(value))
                store.write(staged / "resume.json", snapshot)
                views = render.write(staged, value)
                # Trace is the atomic ownership/history boundary. Remaining
                # views may be stale after interruption and must be republished.
                paths = [staged / "trace.json", staged / "resume.json", *views]
                for source in paths:
                    target = destination / source.relative_to(staged)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source, target)
            index.rebuild(results)
            store.write(directory / "publication.json", dict(
                iteration=state["iterations"] - 1, segment=state["current_measurement_segment"]))
            return dict(directory=str(destination), summary=summary)
