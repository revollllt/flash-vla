"""Environment provenance follows the CUDA device and fails visibly on missing evidence."""
from contextlib import nullcontext
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmarks import latency, metrics
from flash_vla.runtime.identity import MeasurementContext


@pytest.mark.parametrize("uuid", ["1234", "GPU-1234", "MIG-1234"])
def test_selector_uses_current_cuda_device_not_first_visible(uuid, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,3")
    with patch.object(latency.torch.cuda, "current_device", return_value=1), \
         patch.object(latency.torch.cuda, "get_device_properties",
                      return_value=SimpleNamespace(uuid=uuid)) as props:
        assert latency.device_selector() == (uuid if "-" in uuid else "GPU-" + uuid)
    props.assert_called_once_with(1)


def test_environment_stamps_selected_device_power_without_claiming_clock_lock():
    with patch.object(latency, "device_selector", return_value="GPU-selected"), \
         patch.object(latency, "env_block", return_value={"gpu": "H100"}), \
         patch.object(latency.subprocess, "run", return_value=SimpleNamespace(
             stdout="570.86.10, 700.00, 650.00, 1980, 2619\n")) as query:
        env = latency._env()
    argv = query.call_args.args[0]
    assert argv[argv.index("-i") + 1] == "GPU-selected"
    assert query.call_args.kwargs["check"] is True
    assert env["driver"] == "570.86.10"
    assert env["power_policy"] == {"requested_limit_w": 700.0, "enforced_limit_w": 650.0}
    assert env["clock_policy"] is None
    engine = SimpleNamespace(measurement_context={
        "weights": {"checkpoint_id": "a", "checkpoint_digest": "a"},
        "fixture": {"id": "inputs", "digest": "inputs"},
    })
    context = metrics.report_context(engine, env)
    assert context["environment"]["clock_policy"] is None
    assert context["environment"]["power_policy"] == env["power_policy"]
    other = metrics.report_context(engine, dict(env, power_policy={
        "requested_limit_w": 700.0, "enforced_limit_w": 700.0}))
    assert MeasurementContext.from_dict(context).segment_key != MeasurementContext.from_dict(other).segment_key


@pytest.mark.parametrize("stdout", [
    "570.86.10, 700, 700, 1980, 2619\n570.86.10, 500, 500, 1980, 2619\n",
    "570.86.10, N/A, N/A, 1980, 2619\n",
    "",
])
def test_missing_or_ambiguous_power_evidence_is_not_silently_accepted(stdout):
    with patch.object(latency, "device_selector", return_value="GPU-selected"), \
         patch.object(latency, "env_block", return_value={}), \
         patch.object(latency.subprocess, "run", return_value=SimpleNamespace(stdout=stdout)):
        with pytest.raises(ValueError):
            latency._env()


def test_environment_query_error_propagates():
    error = subprocess.CalledProcessError(9, ["nvidia-smi"], stderr="device unavailable")
    with patch.object(latency, "device_selector", return_value="GPU-selected"), \
         patch.object(latency.subprocess, "run", side_effect=error):
        with pytest.raises(subprocess.CalledProcessError) as raised:
            latency._env()
    assert raised.value.stderr == "device unavailable"


@pytest.mark.parametrize("field,before,after", [
    ("power_policy", {"enforced_limit_w": 700}, {"enforced_limit_w": 650}),
    ("clock_observation", {"application_graphics_mhz": "1980"}, {"application_graphics_mhz": "1800"}),
])
def test_last_leg_environment_drift_rejects_complete_run(field, before, after):
    from eval.tests.test_latency import _Engine

    initial = {"driver": "570", field: before}
    changed = {"driver": "570", field: after}
    stats = {"min": 1., "median": 1., "p99": 1.}
    result = dict(chunk_latency=stats, device_latency=stats,
                  host_time={}, segment_latency={}, overhead=stats)
    with patch.object(latency.torch.cuda, "device", return_value=nullcontext()), \
         patch.object(latency, "require_cuda"), \
         patch.object(latency.torch.cuda, "init"), \
         patch.object(latency.torch.cuda, "empty_cache"), \
         patch.object(latency, "resolve", side_effect=lambda value: value), \
         patch.object(latency, "build", side_effect=lambda target, plan, **kwargs: _Engine(plan)), \
         patch.object(latency, "_env", side_effect=[initial] * 5 + [changed]), \
         patch.object(latency, "measure", return_value=result) as measure:
        with pytest.raises(ValueError, match="changed during a leg"):
            latency.run("test", ["a", "b", "a"], reps=1, warmup=0, attribution=False)
    assert measure.call_count == 3

def test_explicit_runner_device_controls_selector_and_gpu_name():
    with patch.object(latency.torch.cuda, "current_device", return_value=0), \
         patch.object(latency.torch.cuda, "get_device_properties",
                      return_value=SimpleNamespace(uuid="other")) as props, \
         patch.object(latency.torch.cuda, "get_device_name", return_value="other GPU") as name:
        assert latency.device_selector("cuda:1") == "GPU-other"
        assert metrics.env_block("cuda:1")["gpu"] == "other GPU"
    props.assert_called_once_with("cuda:1")
    name.assert_called_once_with("cuda:1")

def test_explicit_device_is_used_for_every_latency_leg_and_collector():
    from eval.tests.test_latency import _Engine

    engine = _Engine("a")
    engine.device = "cuda:1"
    stats = {"min": 1., "median": 1., "p99": 1.}
    result = dict(chunk_latency=stats, device_latency=stats,
                  host_time={}, segment_latency={}, overhead=stats)
    with patch.object(latency.torch.cuda, "current_device", return_value=0), \
         patch.object(latency.torch.cuda, "device", return_value=nullcontext()) as device, \
         patch.object(latency, "require_cuda"), \
         patch.object(latency.torch.cuda, "init"), \
         patch.object(latency.torch.cuda, "empty_cache"), \
         patch.object(latency, "resolve", side_effect=lambda value: value), \
         patch.object(latency, "build", return_value=engine), \
         patch.object(latency, "_env", return_value={}) as environment, \
         patch.object(latency, "measure", return_value=result), \
         patch.object(latency, "device_selector", return_value="GPU-other") as selector, \
         patch.object(latency, "attribution_summary", return_value=""), \
         patch.object(latency, "Attribution") as collector:
        collector.return_value.as_dict.return_value = {}
        latency.run("test", ["a"], reps=1, warmup=0, attribution=True, device="cuda:1")
    assert [call.args for call in environment.call_args_list] == [("cuda:1",), ("cuda:1",)]
    device.assert_called_once_with("cuda:1")
    selector.assert_called_once_with("cuda:1")
    collector.assert_called_once_with(device_index="GPU-other")
