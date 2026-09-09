"""Canonical local Campaign discovery. Raw logs are never scanned or copied."""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import shutil
import time
import uuid

from flash_vla.runtime.identity import canonical_digest

from . import campaign, store, transition


def key_digest(key):
    """Only the plan's semantic CampaignKey determines the local location."""
    normalized = campaign.campaign_key(
        dict(schema_version=3, **key["target"], execution_variant=key["execution_variant"]),
        objective=key["objective"], protocol=key["benchmark_protocol"],
    )
    if normalized != key or not key["objective"] or not key["benchmark_protocol"]:
        raise ValueError("expected a canonical CampaignKey without checkpoint or implementation fields")
    return canonical_digest(key).split(":", 1)[1]


class CampaignRegistry:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.directory = self.root / "artifacts/optimization/campaigns"

    @contextmanager
    def _lock(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".registry.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _path(self, key, fork_id=None):
        path = self.directory / key_digest(key)
        if fork_id is not None:
            if str(uuid.UUID(fork_id)) != fork_id:
                raise ValueError("fork ID must be the UUID returned by campaign-fork")
            path = path / "forks" / fork_id
        return path

    def _open(self, key, fork_id=None):
        path = self._path(key, fork_id)
        if not path.exists():
            raise FileNotFoundError(path)
        with (path / "campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            metadata = store.read(path / "campaign.json")
            actual = dict(target=metadata["target"],
                          execution_variant=metadata["execution_variant"],
                          objective=metadata["objective"],
                          benchmark_protocol=metadata["protocol"])
            if actual != key or metadata["id"] != path.name:
                raise ValueError("registry directory and Campaign identity disagree")
            state = campaign.rebuild(path)
            if fork_id is not None:
                provenance = metadata.get("fork", {})
                iteration = provenance.get("parent_iteration")
                if (provenance.get("parent_campaign") != self._path(key).name
                        or not isinstance(iteration, int) or isinstance(iteration, bool)
                        or not 0 <= iteration < state["iterations"]
                        or not isinstance(provenance.get("reason"), str)
                        or not provenance["reason"].strip()
                        or provenance.get("execution_repository") != str(path / "checkout")
):
                    raise ValueError("fork parent provenance is invalid")
        return path

    def find(self, key):
        """Return the validated canonical location, or None only when absent."""
        path = self._path(key)
        if not self.directory.exists():
            return None
        with self._lock():
            return self._open(key) if path.exists() else None

    def open(self, key, *, fork_id=None):
        """Restore derived state under the existing Campaign lock."""
        self._path(key, fork_id)
        with self._lock():
            return self._open(key, fork_id)

    def _create(self, key, baseline, inputs):
        expected = campaign.campaign_key(
            baseline["identity"], objective=key["objective"],
            protocol=key["benchmark_protocol"],
        )
        if expected != key:
            raise ValueError("baseline belongs to another CampaignKey")
        path = self._path(key)
        campaign.create(path, baseline, key["objective"], key["benchmark_protocol"],
                        baseline["measurement_context"]["fixture"]["id"],
                        root=self.root if inputs else None, inputs=inputs)
        store.write(path / "hypotheses.json", {"unresolved": []})
        return path

    def create(self, key, *, baseline, inputs=()):
        """Create one lineage; an existing or partially written directory is an error."""
        self._path(key)
        with self._lock():
            return self._create(key, baseline, inputs)

    def open_or_create(self, key, *, baseline=None, inputs=()):
        """New weights/fixtures discover the existing lineage without changing its context."""
        path = self._path(key)
        with self._lock():
            if path.exists():
                return self._open(key)
            if baseline is None:
                raise FileNotFoundError("no local Campaign; supply validated baseline evidence")
            return self._create(key, baseline, inputs)

    def fork(self, key, *, reason):
        """Explicit local branch of a terminal ledger; no measurement or worker is rerun."""
        if not reason.strip():
            raise ValueError("campaign-fork requires a reason")
        with self._lock():
            parent = self._open(key)
            with (parent / "campaign.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                state = campaign.rebuild(parent)
                records = campaign._records(parent)
                if any(record.get("verdict") is None for _, record in records) or transition.pending(parent):
                    raise RuntimeError("finish or reconcile the active candidate/transition before fork")
                incumbent = campaign.incumbent_record(parent, state)
                if "source" not in incumbent:
                    raise ValueError("fork requires declared portable source inputs")
                child = self._path(key, str(uuid.uuid4()))
                metadata = store.read(parent / "campaign.json")
                metadata.update(id=child.name, created=time.time(),
                                fork=dict(parent_campaign=state["campaign_id"],
                                          parent_iteration=state["iterations"] - 1, reason=reason,
                                          execution_repository=str(child / "checkout")))
                for path, record in records:
                    destination = child / path.relative_to(parent)
                    if "source" in record:
                        record["source"] = _copy_source(record["source"], destination.parent / "inputs")
                    if "directory" in record:
                        record["directory"] = str(destination.parent)
                    store.write(destination, record)
                for path, record in transition.records(parent):
                    # Completed transition reports are retained as historical evidence.
                    # Only their declared source snapshots need independent local copies.
                    destination = child / path.relative_to(parent)
                    if "source" in record:
                        record["source"] = _copy_source(record["source"], destination.parent / "inputs")
                    store.write(destination, record)
                store.write(child / "hypotheses.json", store.read(parent / "hypotheses.json"))
                inherited = campaign.incumbent_record(child, state)
                transition._execution_checkout(dict(
                    directory=str(child), repository=str(self.root), source=inherited["source"],
                    identity=campaign.implementation_identity(inherited),
                    incumbent=state["portable_incumbent"], before_iteration=state["iterations"],
                ))
                store.write(child / "campaign.json", metadata)
                campaign.rebuild(child)
                return child


def _copy_source(source, destination):
    """Retain declared source inputs without copying raw profiler output or checkouts."""
    for name in source["inputs"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(source["root"]) / name, target)
    result = dict(source, root=str(destination))
    store.write(destination / "source.json", result)
    return result
