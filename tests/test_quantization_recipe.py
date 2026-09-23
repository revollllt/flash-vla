"""A quantization recipe is selected at build time, recorded in the identity as
its own workload, and enforced on the routes: its call sites run only its
backends, and no other runner runs them."""
import pytest

from flash_vla.inference import declare

GATED = "llm_backbone_norm_gated_ffn_masked"
DOWN = "llm_backbone_ffn_down_residual_masked"
RECIPE = "mxfp8-llm-ffn"


def test_recipe_routes_its_call_sites_and_is_its_own_workload() -> None:
    bf16 = declare("rtx5090/pi05")
    shipped = declare("rtx5090/pi05", quantization=RECIPE)
    reference = declare("rtx5090/pi05", "reference", quantization=RECIPE)
    assert {shipped.identity.plan[GATED], shipped.identity.plan[DOWN]} == {"mxfp8-backbone"}
    assert {reference.identity.plan[GATED], reference.identity.plan[DOWN]} == {"fake-quant-mxfp8"}
    assert ({name: backend for name, backend in shipped.identity.plan.items()
             if name not in (GATED, DOWN)}
            == {name: backend for name, backend in bf16.identity.plan.items()
                if name not in (GATED, DOWN)})
    quantization = shipped.identity.execution_variant.quantization
    assert quantization["mode"] == shipped.identity.precision == "mxfp8"
    assert quantization["call_sites"] == [DOWN, GATED]
    assert bf16.identity.execution_variant.quantization == {"mode": "bf16"}
    assert shipped.identity.same_workload(reference.identity)
    assert not shipped.identity.same_workload(bf16.identity)


@pytest.mark.parametrize("plan, quantization", [
    ({GATED: "bucketed-backbone", DOWN: "bucketed-backbone"}, RECIPE),   # BF16 kernels
    ({GATED: "mxfp8-backbone", DOWN: "mxfp8-backbone"}, "bf16"),         # unrecorded MXFP8
    ({GATED: "fake-quant-mxfp8", DOWN: "fake-quant-mxfp8"}, "bf16"),
])
def test_misrouted_recipe_is_rejected(plan: dict[str, str], quantization: str) -> None:
    with pytest.raises(ValueError, match="misrouted"):
        declare("rtx5090/pi05", plan, quantization=quantization)


def test_split_mxfp8_ffn_is_rejected() -> None:
    with pytest.raises(ValueError, match="route together"):
        declare("rtx5090/pi05", {GATED: "mxfp8-backbone", DOWN: "fake-quant-mxfp8"},
                quantization=RECIPE)


def test_unknown_recipe_is_rejected() -> None:
    with pytest.raises(KeyError, match="no quantization"):
        declare("rtx5090/pi05", quantization="nvfp4-llm-ffn")


def test_recipe_prices_its_call_sites_in_its_format() -> None:
    dense, quantized = ({invocation.call_site: invocation
                         for invocations in declare("rtx5090/pi05", quantization=q).costs.values()
                         for invocation in invocations} for q in ("bf16", RECIPE))
    rows, width, hidden = 968, 2048, 16384            # prefix rows, model width, FFN width
    per_element = 2 - (1 + 1 / 32)                    # bf16 against MXFP8 with its scales
    saved = {GATED: (2 * width * hidden + rows * hidden) * per_element,    # gate, up, hidden
             DOWN: (hidden * width + rows * hidden) * per_element}         # down, hidden
    for site, bytes_saved in saved.items():
        assert quantized[site].tensor == "mxfp8" and dense[site].tensor == "bf16"
        assert quantized[site].cost.flops == dense[site].cost.flops
        assert abs(dense[site].cost.bytes - quantized[site].cost.bytes - bytes_saved) <= 2
    assert all(quantized[site] == dense[site] for site in dense if site not in saved)
