"""Offline publication validation and deterministic view recovery."""
import fcntl
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from lab.optimize import store
from lab.optimize.registry import key_digest
from . import index, render, resume, schema, validate


def _sources(results):
    paths = sorted(set(index.trace_paths(results)) | {
        p.with_name("trace.json") for p in [*results.glob("targets/*/resume.json"),
                                           *results.glob("targets/*/forks/*/resume.json")]})
    for path in paths:
        if path.with_name("resume.json").exists():
            _, value = resume.reconstruct(store.read(path.with_name("resume.json")))
            if path.exists():
                actual = store.read(path)
                validate.trace(actual)
                if actual != value:
                    raise ValueError("resume snapshot differs from published trace; republish the lineage")
        else:
            value = store.read(path)
        key = validate.trace(value)
        expected = Path("targets") / key_digest(key)
        if "fork" in value:
            fork = value["fork"]
            try:
                fork_id = str(uuid.UUID(value["campaign_id"]))
            except ValueError as error:
                raise ValueError("invalid published fork ID") from error
            if (fork["parent_campaign"] != key_digest(key)
                    or fork_id != value["campaign_id"]
                    or not isinstance(fork["reason"], str) or not fork["reason"].strip()
                    or type(fork["parent_iteration"]) is not int
                    or not 0 <= fork["parent_iteration"] < len(value["iterations"])
                    or not fork["parent_campaign"]):
                raise ValueError("invalid published fork provenance")
            expected /= Path("forks") / value["campaign_id"]
        if path.parent.relative_to(results) != expected:
            raise ValueError("published path does not match CampaignKey/lineage")
        yield path, value


def validate_results(root):
    """Check retained facts and JSON views without rerendering plots."""
    results = Path(root).resolve() / "results"
    if not results.exists():
        return dict(campaigns=0)
    entries = []
    for path, _ in _sources(results):
        summary = schema.check_views(path.parent)
        entries.append(dict(campaign_key=summary["campaign_key"], lineage_id=summary["lineage_id"],
                            summary=(path.parent / "summary.json").relative_to(results).as_posix()))
    if store.read(results / "index.json") != dict(schema_version=1, campaigns=entries):
        raise ValueError("published index is stale relative to traces")
    return dict(campaigns=len(entries))


def rebuild(root, *, check=False):
    """Check or repair derived files; traces remain the published source of facts."""
    root = Path(root).resolve()
    results = root / "results"
    if not results.exists():
        return dict(campaigns=0, changed=[])
    scratch = root / "artifacts/results"
    scratch.mkdir(parents=True, exist_ok=True)
    with (scratch / "publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sources = list(_sources(results))
        with tempfile.TemporaryDirectory(dir=scratch, prefix="rebuild-") as temporary:
            staged = Path(temporary)
            ordered = []
            for path, value in sources:
                destination = staged / path.parent.relative_to(results)
                destination.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    shutil.copyfile(path, destination / "trace.json")
                else:
                    store.write(destination / "trace.json", value)
                ordered.append(destination / "trace.json")
                if path.with_name("resume.json").exists():
                    snapshot, _ = resume.reconstruct(store.read(path.with_name("resume.json")))
                    store.write(destination / "resume.json", snapshot)
                    ordered.append(destination / "resume.json")
                ordered.extend(render.write(destination, value))
            index.rebuild(staged)
            ordered.extend([staged / "index.json", staged / "README.md", staged / "index.html"])
            expected = {path.relative_to(staged): path for path in ordered}
            actual = {path.relative_to(results) for path in results.rglob("*") if path.is_file()}
            unexpected = actual - expected.keys()
            if unexpected:
                raise ValueError(f"unowned published files: {sorted(map(str, unexpected))}")
            changed = [name for name, path in expected.items()
                       if not (results / name).is_file() or (results / name).read_bytes() != path.read_bytes()]
            if check and changed:
                raise ValueError(f"stale published artifacts: {sorted(map(str, changed))}")
            if not check:
                for name in changed:
                    target = results / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(expected[name], target)
            return dict(campaigns=len(sources), changed=sorted(map(str, changed)))
