"""Registry acceptance uses real files, independent processes and existing ledger validation."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from lab.optimize import campaign, store
from lab.optimize.registry import CampaignRegistry, key_digest
from eval.legacy_tests.test_weight_dependency import workspace, candidate, finalize


@pytest.fixture
def setup(workspace):
    root, old = workspace
    baseline = store.read(old / "runs/iter-000-baseline/evidence.json")["measurement"]["evidence"]
    key = campaign.campaign_key(baseline["identity"], objective="e2e_chunk_latency_ms",
                                protocol="latency-v2")
    return root, CampaignRegistry(root), key, baseline


def test_key_and_digest_ignore_checkpoint_fixture_engine_plan_and_order(setup):
    root, registry, key, baseline = setup
    identity = dict(baseline["identity"], engine_revision="another", plan={"different": "kernel"},
                    checkpoint="new", fixture="new", hostname="new", upstream_commit="new")
    other = campaign.campaign_key(identity, objective=key["objective"], protocol=key["benchmark_protocol"])
    assert other == key
    assert key_digest(dict(reversed(list(other.items())))) == key_digest(key)
    assert registry.find(key) is None
    assert not registry.directory.exists()


@pytest.mark.parametrize("field,value", [
    ("objective", "device_latency_ms"), ("benchmark_protocol", "latency-v3"),
    ("execution_variant", {"quantization": {"mode": "fp8"}, "cache": {"mode": "none"}}),
])
def test_semantic_key_axes_separate_campaigns(setup, field, value):
    _, _, key, _ = setup
    assert key_digest(dict(key, **{field: value})) != key_digest(key)


def test_create_restart_and_new_checkpoint_reuse_same_history(setup):
    root, registry, key, baseline = setup
    directory = registry.create(key, baseline=baseline, inputs=["src/kernel.cu"])
    assert directory == root / "artifacts/optimization/campaigns" / key_digest(key)
    campaign.start(root, candidate("rejected"), directory)
    finalize(directory, 1, "no_benefit")
    (directory / "state.json").write_text("stale derived state")
    new_baseline = deepcopy(baseline)
    new_baseline["measurement_context"]["weights"]["checkpoint_digest"] = "checkpoint-b"
    resumed = CampaignRegistry(root).open_or_create(key, baseline=new_baseline)
    assert resumed == directory
    state = store.read(resumed / "state.json")
    assert state["iterations"] == 2
    assert state["failed_hypotheses"][0]["candidate_id"] == "rejected"
    assert state["measurement_context"] == baseline["measurement_context"]
    assert registry.find(key) == registry.open(key) == directory
    campaign.start(root, candidate("next"), resumed)
    assert campaign.rebuild(resumed)["iterations"] == 3


def test_duplicate_create_and_mismatched_baseline_refused(setup):
    _, registry, key, baseline = setup
    bad = dict(key, objective="device_latency_ms")
    with pytest.raises(ValueError, match="objective"):
        registry.create(bad, baseline=baseline)
    assert registry.find(bad) is None
    directory = registry.create(key, baseline=baseline)
    with pytest.raises(FileExistsError):
        registry.create(key, baseline=baseline)
    assert campaign.rebuild(directory)["iterations"] == 1


@pytest.mark.parametrize("corruption", ["metadata", "missing_baseline", "key", "iteration"])
def test_corrupt_authoritative_campaign_is_not_replaced(setup, corruption):
    _, registry, key, baseline = setup
    directory = registry.create(key, baseline=baseline)
    path = directory / "campaign.json"
    if corruption == "metadata":
        path.write_text("{")
    elif corruption == "key":
        value = store.read(path)
        value["target"]["model_revision"] = "wrong"
        store.write(path, value)
    else:
        path = directory / "runs/iter-000-baseline/evidence.json"
        if corruption == "missing_baseline":
            path.unlink()
        else:
            value = store.read(path)
            value["iteration"] = 9
            store.write(path, value)
    before = path.read_text() if path.exists() else None
    for action in (registry.find, registry.open, registry.open_or_create):
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            action(key)
    assert (path.read_text() if path.exists() else None) == before


def test_two_fresh_processes_create_only_one_lineage(setup, tmp_path):
    root, _, key, baseline = setup
    spec = tmp_path / "input.json"
    store.write(spec, dict(key=key, baseline=baseline))
    barrier = tmp_path / "go"
    script = """
import json, pathlib, sys, time
from lab.optimize.registry import CampaignRegistry
spec = json.loads(pathlib.Path(sys.argv[2]).read_text())
pathlib.Path(sys.argv[3] + "." + sys.argv[4]).touch()
while not pathlib.Path(sys.argv[3]).exists():
    time.sleep(.01)
print(CampaignRegistry(sys.argv[1]).open_or_create(spec["key"], baseline=spec["baseline"]))
"""
    processes = [subprocess.Popen([sys.executable, "-c", script, str(root), str(spec), str(barrier), str(i)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for i in range(2)]
    import time
    deadline = time.monotonic() + 20
    while len(list(tmp_path.glob("go.*"))) < 2 and time.monotonic() < deadline:
        time.sleep(.01)
    assert len(list(tmp_path.glob("go.*"))) == 2
    barrier.touch()
    outputs = [process.communicate(timeout=30) for process in processes]
    assert [process.returncode for process in processes] == [0, 0], outputs
    assert outputs[0][0] == outputs[1][0]
    directory = Path(outputs[0][0].strip())
    assert len(list(directory.parent.glob("*/campaign.json"))) == 1
    assert campaign.rebuild(directory)["iterations"] == 1


def test_fork_keeps_history_source_and_parent_but_never_copies_raw_logs(setup):
    root, registry, key, baseline = setup
    parent = registry.create(key, baseline=baseline, inputs=["src/kernel.cu"])
    campaign.start(root, candidate("rejected"), parent)
    finalize(parent, 1, "no_benefit")
    store.write(parent / "hypotheses.json", {"unresolved": [{"mechanism": "next mechanism"}]})
    (parent / "runs/iter-001-rejected/check.stdout").write_text("large raw log")
    child = registry.fork(key, reason="compare another kernel strategy")
    metadata = store.read(child / "campaign.json")
    assert metadata["fork"]["parent_campaign"] == parent.name
    assert metadata["fork"]["parent_iteration"] == 1
    assert metadata["fork"]["reason"] == "compare another kernel strategy"
    assert metadata["fork"]["execution_repository"] == str(child / "checkout")
    assert registry.open(key, fork_id=child.name) == child
    assert registry.open(key) == parent
    assert not (child / "runs/iter-001-rejected/check.stdout").exists()
    source = store.read(child / "runs/iter-000-baseline/evidence.json")["source"]
    assert Path(source["root"]).is_relative_to(child)
    assert (Path(source["root"]) / "src/kernel.cu").read_text() == "original"
    assert store.read(child / "hypotheses.json") == store.read(parent / "hypotheses.json")
    campaign.start(root, candidate("alternative"), child)
    assert campaign.rebuild(child)["iterations"] == 3
    assert campaign.rebuild(parent)["iterations"] == 2


def test_fork_refuses_unfinished_work_and_empty_reason(setup):
    root, registry, key, baseline = setup
    directory = registry.create(key, baseline=baseline, inputs=["src/kernel.cu"])
    with pytest.raises(ValueError, match="reason"):
        registry.fork(key, reason=" ")
    campaign.start(root, candidate(), directory)
    with pytest.raises(RuntimeError, match="active candidate"):
        registry.fork(key, reason="alternative")
    assert not (directory / "forks").exists()


def test_cli_creates_and_opens_without_human_campaign_directory(setup, tmp_path):
    root, _, key, baseline = setup
    evidence = tmp_path / "baseline.json"
    store.write(evidence, baseline)
    key_path = tmp_path / "key.json"
    store.write(key_path, key)
    command = [sys.executable, "-m", "lab.optimize"]
    created = subprocess.run(command + [
        "campaign-create", "--root", str(root), "--baseline-evidence", str(evidence),
        "--objective", key["objective"], "--protocol", key["benchmark_protocol"],
        "--fixture", baseline["measurement_context"]["fixture"]["id"], "--source-input", "src/kernel.cu"],
        capture_output=True, text=True, check=True)
    location = json.loads(created.stdout)["directory"]
    for action in ("campaign-find", "campaign-open", "campaign-open-or-create"):
        opened = subprocess.run(command + [action, str(key_path), "--root", str(root)],
                                capture_output=True, text=True, check=True)
        assert json.loads(opened.stdout)["directory"] == location

@pytest.mark.parametrize("field,value", [
    ("parent_campaign", "missing-parent"), ("parent_iteration", -1),
    ("parent_iteration", 1000), ("reason", ""), ("parent_iteration", True),
    ("execution_repository", "/another/checkout"),
])
def test_corrupt_fork_provenance_is_rejected(setup, field, value):
    _, registry, key, baseline = setup
    registry.create(key, baseline=baseline, inputs=["src/kernel.cu"])
    child = registry.fork(key, reason="alternative")
    path = child / "campaign.json"
    metadata = store.read(path)
    metadata["fork"][field] = value
    store.write(path, metadata)
    with pytest.raises(ValueError, match="fork parent provenance"):
        registry.open(key, fork_id=child.name)


def test_fork_after_transition_has_independent_execution_checkout(setup):
    from eval.legacy_tests.test_context_transition import request, next_spec

    root, registry, key, baseline = setup
    parent = registry.create(key, baseline=baseline, inputs=["src/kernel.cu"])
    parent_state = campaign.transition_context(root, parent, request(root, parent))
    child = registry.fork(key, reason="new strategy on checkpoint B")
    child_state = campaign.rebuild(child)
    execution = Path(child_state["execution_repository"])
    assert execution != Path(parent_state["execution_repository"])
    assert execution.is_relative_to(child)
    assert (execution / "src/kernel.cu").read_text() == "original"
    actual = subprocess.check_output(["git", "-C", str(execution), "rev-parse", "HEAD"], text=True).strip()
    assert actual == baseline["identity"]["engine_revision"]
    (execution / "src/kernel.cu").write_text("child strategy")
    assert (Path(parent_state["execution_repository"]) / "src/kernel.cu").read_text() == "original"
    campaign.start(execution, next_spec(child), child)
    assert campaign.rebuild(parent)["iterations"] == 1
    assert campaign.rebuild(child)["iterations"] == 2

def test_failed_fork_checkout_is_not_published_as_openable(setup, monkeypatch):
    from lab.optimize import transition

    _, registry, key, baseline = setup
    parent = registry.create(key, baseline=baseline, inputs=["src/kernel.cu"])

    def fail(record):
        raise RuntimeError("injected worktree creation failure")

    monkeypatch.setattr(transition, "_execution_checkout", fail)
    with pytest.raises(RuntimeError, match="injected"):
        registry.fork(key, reason="alternative")
    incomplete = next((parent / "forks").iterdir())
    assert not (incomplete / "campaign.json").exists()
    with pytest.raises(FileNotFoundError):
        registry.open(key, fork_id=incomplete.name)
    assert registry.open(key) == parent


def test_child_transition_uses_its_new_checkout_after_fork(setup):
    from eval.legacy_tests.test_context_transition import request

    root, registry, key, baseline = setup
    parent = registry.create(key, baseline=baseline, inputs=["src/kernel.cu"])
    campaign.transition_context(root, parent, request(root, parent))
    child = registry.fork(key, reason="alternative")
    incoming = request(root, child)
    incoming["measurement_context"]["weights"]["checkpoint_digest"] = "checkpoint-c"
    state = campaign.transition_context(root, child, incoming)
    execution = Path(state["execution_repository"])
    assert execution.is_relative_to(child / "transitions")
    assert state["current_measurement_segment"] == 2
    assert registry.open(key, fork_id=child.name) == child
    assert campaign.rebuild(parent)["current_measurement_segment"] == 1


def test_two_processes_finalize_one_iteration_atomically(workspace):
    import fcntl

    root, directory = workspace
    campaign.start(root, candidate("no-op"), directory)
    worker = """
import sys
from pathlib import Path
from eval.legacy_tests.test_weight_dependency import finalize
print("ready", flush=True)
try:
    finalize(Path(sys.argv[1]), 1, "no_benefit",
             result={"validity": "valid", "candidate_ms": 16.0})
except ValueError as error:
    if str(error) != "iteration verdict is immutable":
        raise
    print("already finalized", flush=True)
else:
    print("finalized", flush=True)
"""
    processes = []
    try:
        with (directory / "campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for _ in range(2):
                processes.append(subprocess.Popen(
                    [sys.executable, "-c", worker, str(directory)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
            for process in processes:
                assert process.stdout.readline().strip() == "ready"
        outputs = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stderr
            outputs.append(stdout.strip())
        assert sorted(outputs) == ["already finalized", "finalized"]
        state = campaign.rebuild(directory)
        assert state["current_incumbent"] == "iter-000"
        assert state["iterations"] == 2
        assert len(list((directory / "runs").glob("iter-001-*/evidence.json"))) == 1
        assert campaign.start(root, candidate("next"), directory)["iteration"] == 2
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_measurement_interrupt_is_discovered_and_explicitly_resumed(setup):
    import signal
    import time

    root, registry, key, baseline = setup
    directory = registry.create(key, baseline=baseline, inputs=["src/kernel.cu"])
    marker = root / "measurement-starts"
    measurement = """
from pathlib import Path
import sys, time
marker = Path(sys.argv[1])
with marker.open("a") as output:
    print("started", file=output)
if len(marker.read_text().splitlines()) == 1:
    time.sleep(30)
"""
    spec = candidate("interrupted")
    spec["stages"] = {
        "check": dict(argv=[sys.executable, "-c", "print('pass')"],
                      resource="cpu", timeout_s=10),
        "measure": dict(argv=[sys.executable, "-c", measurement, str(marker)],
                        resource="cpu", timeout_s=40),
    }
    record = campaign.start(root, spec, directory)
    worker = subprocess.Popen(
        [sys.executable, "-c",
         "import sys; from lab.optimize import campaign; campaign.resume(sys.argv[1])",
         str(directory)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and worker.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), "measurement did not start"
        worker.send_signal(signal.SIGINT)
        _, stderr = worker.communicate(timeout=10)
        assert worker.returncode != 0 and "KeyboardInterrupt" in stderr
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()
    interrupted = store.read(Path(record["directory"]) / "evidence.json")
    assert interrupted["stages"]["measure"]["status"] == "interrupted"
    recovery = """
import json, sys
from pathlib import Path
from lab.optimize import campaign, store
from lab.optimize.registry import CampaignRegistry
from eval.legacy_tests.test_weight_dependency import candidate, finalize
root, key = Path(sys.argv[1]), json.loads(sys.argv[2])
directory = CampaignRegistry(root).open(key)
marker = root / "measurement-starts"
try:
    campaign.resume(directory)
except RuntimeError as error:
    assert "explicit reconcile" in str(error)
else:
    raise AssertionError("silently resumed interrupted measurement")
assert len(marker.read_text().splitlines()) == 1
state = campaign.resume(directory, recovered_seconds=0)
assert len(marker.read_text().splitlines()) == 2
record = store.read(next((directory / "runs").glob("iter-001-*/evidence.json")))
assert len(record["attempts"]) == 1
assert record["stages"]["check"]["status"] == "completed"
assert record["stages"]["measure"]["status"] == "completed"
finalize(directory, 1, "no_benefit",
         result={"validity": "valid", "candidate_ms": 16.0})
assert campaign.start(root, candidate("next"), directory)["iteration"] == 2
print(json.dumps({"directory": str(directory), "archived_attempts": len(record["attempts"]),
                  "measurement_starts": 2, "next_iteration": 2}))
"""
    result = subprocess.run(
        [sys.executable, "-c", recovery, str(root), json.dumps(key)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["directory"] == str(directory)
