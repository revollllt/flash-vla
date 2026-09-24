"""The end-to-end model references, checked without weights or a GPU.

Each reference names its parameters as the official checkpoint does, so the
schema it builds on the meta device is the official schema: converting that
schema must give exactly the engine's weight layout, binding it must reproduce
upstream's inference dtypes, and a forward on meta tensors must produce every
stage output in its documented shape. Numerical agreement with the engine is
`eval.model_reference`, on a GPU.
"""
import torch

from flash_vla.models.official import official_schema


def test_pi05_official_schema_converts_to_the_engine_layout():
    from flash_vla.models.pi05 import openpi, reference, spec

    schema = official_schema(reference.make_reference().parts(), prefixes=reference.PREFIXES)
    converted = openpi.target_checkpoint(
        {name: torch.empty(shape, device="meta") for name, shape in schema.items()})
    assert {name: tuple(tensor.shape) for name, tensor in converted.items()} == spec.weight_shapes()


def test_pi05_reference_keeps_upstreams_float32_parameters():
    """OpenPI runs PaliGemma-with-expert in bfloat16 but keeps the SigLIP
    embeddings, every RMSNorm (for the expert, its adaptive dense layers) and
    the model's own heads in float32."""
    from flash_vla.models.pi05 import reference

    prefixes = reference.PREFIXES
    schema = official_schema(reference.make_reference().parts(), prefixes=prefixes)
    model = reference.load({name: torch.empty(shape, dtype=torch.bfloat16, device="meta")
                            for name, shape in schema.items()})
    float32 = {prefixes[part] + name for part, module in model.parts().items()
               for name, parameter in module.named_parameters() if parameter.dtype == torch.float32}
    norms = ("input_layernorm", "post_attention_layernorm")
    backbone, expert = prefixes["backbone"], prefixes["expert"]
    expected = {
        *(prefixes["vision"] + f"embeddings.{name}" for name in (
            "patch_embedding.weight", "patch_embedding.bias", "position_embedding.weight")),
        *(f"{backbone}layers.{layer}.{norm}.weight" for layer in range(18) for norm in norms),
        f"{backbone}norm.weight",
        *(f"{expert}layers.{layer}.{norm}.dense.{kind}"
          for layer in range(18) for norm in norms for kind in ("weight", "bias")),
        f"{expert}norm.dense.weight", f"{expert}norm.dense.bias",
        *(f"{head}.{kind}" for head in ("action_in_proj", "action_out_proj", "time_mlp_in",
                                        "time_mlp_out") for kind in ("weight", "bias")),
    }
    assert float32 == expected and len(expected) == 3 + 37 + 74 + 8


def test_pi05_reference_produces_every_stage_on_meta():
    from flash_vla.models.pi05 import reference

    meta = torch.device("meta")
    outputs = reference.make_reference()(
        torch.empty(3, 224, 224, 3, device=meta), image_masks=torch.ones(3, dtype=torch.bool, device=meta),
        prompt_ids=torch.zeros(200, dtype=torch.long, device=meta),
        prompt_mask=torch.ones(200, dtype=torch.bool, device=meta),
        noise=torch.empty(50, 32, device=meta), steps=2, depth=2)
    assert outputs.vision_hidden.shape == (3, 256, 1152)
    assert [tuple(key.shape) for key, _ in outputs.prefix.cache] == [(1, 1, 968, 256)] * 2
    assert [tuple(key.shape) for key, _ in outputs.suffix_cache] == [(1, 1, 50, 256)] * 2
    assert outputs.actions.shape == (50, 32)


def test_pi0_official_schema_converts_to_the_engine_layout():
    from flash_vla.models.pi0 import openpi, reference, spec

    schema = official_schema(reference.make_reference().parts(), prefixes=reference.PREFIXES)
    converted = openpi.target_checkpoint(
        {name: torch.empty(shape, device="meta") for name, shape in schema.items()},
        prompt_ids=torch.zeros(7, dtype=torch.long, device="meta"))
    assert ({name: tuple(tensor.shape) for name, tensor in converted.items()}
            == spec.weight_shapes(prompt_len=7))


def test_pi0_reference_keeps_upstreams_float32_parameters():
    """Pi0's expert norms are plain RMSNorms, kept float32 like the backbone's."""
    from flash_vla.models.pi0 import reference

    prefixes = reference.PREFIXES
    schema = official_schema(reference.make_reference().parts(), prefixes=prefixes)
    model = reference.load({name: torch.empty(shape, dtype=torch.bfloat16, device="meta")
                            for name, shape in schema.items()})
    float32 = {prefixes[part] + name for part, module in model.parts().items()
               for name, parameter in module.named_parameters() if parameter.dtype == torch.float32}
    norms = ("input_layernorm", "post_attention_layernorm")
    expected = {
        *(prefixes["vision"] + f"embeddings.{name}" for name in (
            "patch_embedding.weight", "patch_embedding.bias", "position_embedding.weight")),
        *(f"{prefixes[stack]}layers.{layer}.{norm}.weight"
          for stack in ("backbone", "expert") for layer in range(18) for norm in norms),
        f"{prefixes['backbone']}norm.weight", f"{prefixes['expert']}norm.weight",
        *(f"{head}.{kind}" for head in ("state_proj", "action_in_proj", "action_time_mlp_in",
                                        "action_time_mlp_out", "action_out_proj")
          for kind in ("weight", "bias")),
    }
    assert float32 == expected and len(expected) == 3 + 37 + 37 + 10


def test_pi0_reference_produces_every_stage_on_meta():
    from flash_vla.models.pi0 import reference

    meta = torch.device("meta")
    outputs = reference.make_reference()(
        torch.empty(3, 224, 224, 3, device=meta), image_masks=torch.ones(3, dtype=torch.bool, device=meta),
        prompt_ids=torch.zeros(0, dtype=torch.long, device=meta),
        prompt_mask=torch.ones(0, dtype=torch.bool, device=meta), state=torch.empty(32, device=meta),
        noise=torch.empty(50, 32, device=meta), steps=2, depth=2)
    assert outputs.vision_hidden.shape == (3, 256, 1152)
    assert [tuple(key.shape) for key, _ in outputs.prefix.cache] == [(1, 1, 768, 256)] * 2
    assert [tuple(key.shape) for key, _ in outputs.suffix_cache] == [(1, 1, 51, 256)] * 2
    assert outputs.actions.shape == (50, 32)
