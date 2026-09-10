from copy import deepcopy
import pytest
from eval.tests.test_checkpoint_compatibility import producer_report


def transition_context(report):
    return dict(weights=deepcopy(report["weights"]), fixture=deepcopy(report["fixture"]),
                environment=dict(gpu_sku="cpu-test", driver="test", cuda_runtime="test",
                                 pytorch="test", tilelang="test", clock_policy="test",
                                 power_policy="test", capture_regime="test"),
                hostname="cpu-test", slurm_job_id="cpu-test", timestamp=1, reference_provenance={})


def test_report_is_consumed_by_onboarding_and_transition(producer_report):
    from lab import onboarding
    from lab.optimize import measurement, reports
    minimal_spec = {"target": {"inference_signature": producer_report["identity"]["inference_signature"]},
                    "initial_weights": producer_report["weights"]}
    onboarding._validate_stage("weights_compatibility", producer_report, minimal_spec)
    context = transition_context(producer_report)
    receipt = reports.compatibility(producer_report, context=context)
    measurement.compatibility(receipt, producer_report["identity"], context)
    assert receipt["status"] == "pass"
    assert receipt["measurement_context"] == context


@pytest.mark.parametrize("change", ["signature", "weights", "fixture", "status"])
def test_transition_cannot_attach_structural_report_to_other_assets(producer_report, change):
    from lab.optimize import reports
    context = transition_context(producer_report)
    if change == "signature":
        producer_report["contract"]["control_flow"]["state"] = "continuous"
    elif change == "status":
        producer_report["status"] = "failed"
    elif change == "weights":
        context["weights"]["checkpoint_digest"] = "other"
    else:
        context["fixture"]["digest"] = "other"
    with pytest.raises(ValueError):
        reports.compatibility(producer_report, context=context)
