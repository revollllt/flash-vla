"""Checkpoint transitions: durable commands and atomic context activation.

Transition records are outside optimization iterations. Completed command receipts
are reused; an unfinished attempt requires explicit reconciliation.
"""
import fcntl
from pathlib import Path
import time
import subprocess

from . import measurement, runner, store
from .schema import validate_applicability, validate_command


def records(directory):
    return [(path, store.read(path)) for path in
            sorted((Path(directory) / "transitions").glob("transition-*/evidence.json"))]


def pending(directory):
    return [(path, record) for path, record in records(directory)
            if record["status"] not in ("activated", "aborted")]


def segments(directory, baseline):
    """Return only activated segment facts, including the original anchor."""
    result = [dict(id=0, before_iteration=0, incumbent="iter-000",
                   measurement=baseline, transition=None)]
    for path, record in records(directory):
        if record["status"] == "activated":
            segment = record["segment"]
            if segment["id"] != len(result):
                raise ValueError("measurement segment IDs must be contiguous")
            result.append(dict(segment, transition=str(path.parent)))
    return result


def descriptor(segment, protocol):
    return dict(id=segment["id"], benchmark_protocol=protocol,
                **segment["measurement"]["measurement_context"]["environment"])


def _commands(request, inherited):
    commands = {"compatibility": request["compatibility"]}
    # Every transition has a fresh execution/output directory. Equal asset IDs
    # do not establish that its required derived artifacts already exist.
    for dependency, field in (("rebuild", "artifact_recipe"), ("retune", "retune_recipe")):
        for record in inherited:
            if validate_applicability(record["spec"]) == dependency:
                commands[f"{dependency}-iter-{record['iteration']:03d}"] = record["spec"][field]
    commands.update(check=request["check"], measure=request["measure"])
    for name, command in commands.items():
        validate_command(name, command)
    return commands


def begin(root, directory, request):
    """Reserve a transition after workload/protocol checks, before any subprocess."""
    from . import campaign

    root, directory = Path(root).resolve(), Path(directory).resolve()
    with (directory / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = store.read(directory / "campaign.json")
        if "execution_variant" not in metadata:
            raise ValueError("context transition requires an explicitly migrated v3 campaign")
        state = campaign.rebuild(directory)
        if pending(directory):
            raise RuntimeError("context transition already pending; resume or reconcile it")
        if any(record.get("verdict") is None for _, record in campaign._records(directory)):
            raise RuntimeError("cannot transition during an active optimization candidate")
        incumbent = campaign.incumbent_record(directory, state)
        identity = campaign.implementation_identity(incumbent)
        measurement.identity(request["identity"], identity)
        if request["protocol"] != metadata["protocol"]:
            raise ValueError("protocol change requires a new Campaign")
        if request["objective"] != metadata["objective"]:
            raise ValueError("objective change requires a new Campaign")
        measurement.context(request["measurement_context"])
        inherited = campaign.portable_optimizations(directory)
        commands = _commands(request, inherited)
        if "source" not in incumbent:
            raise ValueError("portable incumbent source snapshot is required for transition")
        number = len(records(directory))
        run_dir = directory / "transitions" / f"transition-{number:03d}"
        spec = dict(id=run_dir.name, task_id=metadata["id"], target=identity["target"],
                    kind="control", protocol=metadata["protocol"], stages=commands,
                    hypothesis=dict(mechanism="incumbent context re-anchor"),
                    conditions=request["measurement_context"])
        record = dict(id=run_dir.name, target=identity["target"], kind="context_transition",
                      directory=str(run_dir), repository=str(root), source=incumbent["source"],
                      spec=spec, request=request, identity=identity, stages={},
                      before_iteration=state["iterations"], incumbent=state["portable_incumbent"],
                      segment_id=state["current_measurement_segment"] + 1,
                      inherited_iterations=[item["iteration"] for item in inherited],
                      status="ready", validity="incomplete", correctness="not_run",
                      conclusion="inconclusive", promotion="not_requested",
                      cost=dict(cpu_seconds=0.0, gpu_seconds=0.0, jobs=[]),
                      budget=metadata["budget"], timestamp=time.time())
        store.write(run_dir / "request.json", request)
        store.write(run_dir / "evidence.json", record)
        campaign.rebuild(directory)
        return run_dir


def _execution_checkout(record):
    """Use the incumbent commit in an isolated clean checkout for real runner identity."""
    directory = Path(record["directory"]) / "checkout"
    source = record["source"]
    revision = source["revision"]
    if revision != record["identity"]["engine_revision"]:
        raise ValueError("incumbent source revision differs from recorded engine")
    if not directory.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(directory), revision],
                       cwd=record["repository"], check=True, capture_output=True, text=True)
    for name in source["inputs"]:
        if (directory / name).read_bytes() != (Path(source["root"]) / name).read_bytes():
            raise ValueError("portable snapshot differs from its committed engine; commit and requalify")
    observed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=directory, text=True)
    if observed != revision or dirty:
        raise ValueError("transition execution checkout no longer matches the clean incumbent")
    return dict(repository=str(directory), incumbent=record["incumbent"],
                after_iteration=record["before_iteration"] - 1, source=source,
                identity=record["identity"], timestamp=time.time())


def _validate_output(record, name):
    request = record["request"]
    if name in ("compatibility", "check"):
        report = store.read(Path(record["directory"]) / f"{name}.stdout")
        validator = measurement.compatibility if name == "compatibility" else measurement.correctness
        validator(report, record["identity"], request["measurement_context"])
    elif name == "measure":
        report = store.read(Path(record["directory"]) / "measure.stdout")
        measurement.latency(report, expected_identity=record["identity"],
                            expected_context=request["measurement_context"],
                            protocol=request["protocol"], objective=request["objective"])


def run(directory, run_dir):
    """Execute actual recorded commands, then atomically publish the new segment."""
    from . import campaign

    directory, run_dir = Path(directory).resolve(), Path(run_dir).resolve()
    with (run_dir / "worker.lock").open("a") as worker:
        fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = store.read(run_dir / "evidence.json")
        if record["status"] == "activated":
            return campaign.rebuild(directory)
        if record["status"] == "aborted":
            raise RuntimeError("transition was aborted")
        if any(stage["status"] != "completed" for stage in record["stages"].values()):
            raise RuntimeError("unfinished transition requires explicit reconcile before resume")
        for name in record["spec"]["stages"]:
            if name != "compatibility" and "materialization" not in record:
                with (directory / "campaign.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    campaign.rebuild(directory)
                    receipt = _execution_checkout(record)
                    record["materialization"] = receipt
                    record["repository"] = receipt["repository"]
                    store.write(directory / "materializations" /
                                f'after-iter-{receipt["after_iteration"]:03d}.json', receipt)
                    store.write(run_dir / "evidence.json", record)
            if record["stages"].get(name, {}).get("status") != "completed":
                runner.run_stage(record, name)
                if record["stages"][name]["status"] != "completed":
                    raise RuntimeError(f"transition {name} failed; reconcile before retry")
            try:
                _validate_output(record, name)
            except (ValueError, KeyError, OSError) as error:
                record["stages"][name].update(status="failed", validation_error=str(error))
                record.update(status="stopped", validity="incomplete")
                store.write(run_dir / "evidence.json", record)
                raise
        anchor = store.read(run_dir / "measure.stdout")
        anchor["correctness"] = store.read(run_dir / "check.stdout")
        with (directory / "campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = campaign.rebuild(directory)
            if (state["iterations"] != record["before_iteration"]
                    or state["portable_incumbent"] != record["incumbent"]):
                raise RuntimeError("campaign changed while transition was pending")
            record["segment"] = dict(id=record["segment_id"],
                                     before_iteration=record["before_iteration"],
                                     incumbent=record["incumbent"], measurement=anchor,
                                     repository=record["repository"])
            record.update(status="activated", validity="valid", conclusion="reanchor")
            store.write(run_dir / "evidence.json", record)
            return campaign.rebuild(directory)


def resume(directory, *, recovered_seconds=None):
    """Resume a single pending transition; retries require explicit accounting."""
    directory = Path(directory).resolve()
    active = pending(directory)
    if len(active) != 1:
        raise ValueError("expected one pending context transition")
    path, record = active[0]
    if any(stage["status"] != "completed" for stage in record["stages"].values()):
        if recovered_seconds is None:
            raise RuntimeError("unfinished transition requires explicit reconcile before resume")
        runner.reconcile(path.parent, recovered_seconds)
    return run(directory, path.parent)


def abort(directory, *, recovered_seconds=None):
    """Stop a transition after proving its worker stopped; re-anchor remains required."""
    from . import campaign

    directory = Path(directory).resolve()
    active = pending(directory)
    if len(active) != 1:
        raise ValueError("expected one pending context transition")
    path, record = active[0]
    runner.reconcile(path.parent, recovered_seconds)
    with (path.parent / "worker.lock").open("a") as worker:
        fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (directory / "campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            record = store.read(path)
            if record["status"] in ("activated", "aborted"):
                raise RuntimeError("transition already terminal")
            record.update(status="aborted", conclusion="aborted")
            store.write(path, record)
            return campaign.rebuild(directory)
