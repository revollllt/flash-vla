"""The end-to-end model references, checked without weights or a GPU.

Each reference names its parameters as the official checkpoint does, so the
schema it builds on the meta device is the official schema: converting that
schema must give exactly the engine's weight layout, binding it must reproduce
upstream's inference dtypes, and a forward on meta tensors must produce every
stage output in its documented shape. Numerical agreement with the engine is
`eval.model_reference`, on a GPU.
"""
from importlib import import_module

import pytest
import torch

from flash_vla.inference import TARGETS
from flash_vla.models.official import official_schema


def test_every_target_pairs_its_engine_with_the_reference() -> None:
    """Each registered Target names the reference view and the official
    weights `eval.model_reference` runs it against."""
    for entry in TARGETS.values():
        view = import_module(entry.reference_view_module)
        assert callable(view.reference_outputs) and callable(view.comparable)
        assert callable(import_module(entry.sources_module).official_weights)


def test_pi05_official_schema_converts_to_the_engine_layout() -> None:
    from flash_vla.models.pi05 import openpi, reference, spec

    schema = official_schema(reference.make_reference().parts(), prefixes=reference.PREFIXES)
    converted = openpi.target_checkpoint(
        {name: torch.empty(shape, device="meta") for name, shape in schema.items()})
    assert {name: tuple(tensor.shape) for name, tensor in converted.items()} == spec.weight_shapes()


def test_pi05_reference_keeps_upstreams_float32_parameters() -> None:
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


def test_pi05_reference_produces_every_stage_on_meta() -> None:
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


def test_pi0_official_schema_converts_to_the_engine_layout() -> None:
    from flash_vla.models.pi0 import openpi, reference, spec

    schema = official_schema(reference.make_reference().parts(), prefixes=reference.PREFIXES)
    converted = openpi.target_checkpoint(
        {name: torch.empty(shape, device="meta") for name, shape in schema.items()},
        prompt_ids=torch.zeros(7, dtype=torch.long, device="meta"))
    assert ({name: tuple(tensor.shape) for name, tensor in converted.items()}
            == spec.weight_shapes(prompt_len=7))


def test_pi0_reference_keeps_upstreams_float32_parameters() -> None:
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


def test_pi0_reference_produces_every_stage_on_meta() -> None:
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


def test_groot_reference_reads_every_official_tensor_but_the_language_head() -> None:
    """The two LIBERO shards hold 494 backbone and 537 action-head tensors;
    the reference holds all of them but `lm_head` (and runs all but the
    language model's final norm), under the names Transformers' Qwen3-VL and
    Diffusers give them."""
    from flash_vla.models.groot_n17 import reference

    prefixes = reference.PREFIXES
    schema = official_schema(reference.make_reference().parts(), prefixes=prefixes)
    counts = {part: sum(name.startswith(prefix) for name in schema)
              for part, prefix in prefixes.items()}
    assert counts == {"vision": 315, "backbone": 178, "action": 537}
    vision, backbone = prefixes["vision"], prefixes["backbone"]
    block = "action_head.model.transformer_blocks.0."
    expected = {
        f"{vision}patch_embed.proj.weight": (1024, 3, 2, 16, 16),
        f"{vision}merger.norm.weight": (1024,),
        f"{vision}deepstack_merger_list.2.norm.weight": (4096,),
        f"{backbone}layers.15.self_attn.k_norm.weight": (128,),
        f"{block}attn1.to_k.weight": (1536, 2048),
        f"{block}attn1.to_out.0.weight": (1536, 1536),
        f"{block}ff.net.0.proj.weight": (6144, 1536),
        f"{block}ff.net.2.weight": (1536, 6144),
        "action_head.model.timestep_encoder.timestep_embedder.linear_1.weight": (1536, 256),
    }
    assert {name: schema[name] for name in expected} == expected


def test_groot_reference_casts_its_rope_buffers_with_the_model() -> None:
    """GR00T's policy casts the whole model to bfloat16, RoPE's inverse
    frequencies included; float32 keeps every parameter and buffer float32."""
    from flash_vla.models.groot_n17 import reference

    schema = official_schema(reference.make_reference().parts(), prefixes=reference.PREFIXES)
    weights = {name: torch.empty(shape, dtype=torch.bfloat16, device="meta")
               for name, shape in schema.items()}
    for precision, dtype in (("bfloat16", torch.bfloat16), ("float32", torch.float32)):
        model = reference.load(weights, precision=precision)
        assert {tensor.dtype for tensor in (*model.parameters(), *model.buffers())} == {dtype}
        assert {model.vision.inverse_frequency.dtype, model.backbone.inverse_frequency.dtype} == {dtype}


@pytest.mark.parametrize("workload", ["robodojo", "libero"])
def test_groot_reference_produces_every_stage_on_meta(workload: str) -> None:
    from flash_vla.models.groot_n17 import reference
    from flash_vla.models.groot_n17.definition import GrootModel
    from flash_vla.models.groot_n17.spec import PATCH_WIDTH, PATCHES_PER_VIEW, VIEW_GRID, VISUAL_TOKENS_PER_VIEW

    options = GrootModel().workload(workload).options
    views, length = options["views"], options["sequence_length"]
    visual_tokens = views * VISUAL_TOKENS_PER_VIEW
    meta = torch.device("meta")
    schema = official_schema(reference.make_reference().parts(), prefixes=reference.PREFIXES)
    model = reference.load({name: torch.empty(shape, dtype=torch.bfloat16, device=meta)
                            for name, shape in schema.items()})
    outputs = model(
        torch.empty(views * PATCHES_PER_VIEW, PATCH_WIDTH, dtype=torch.bfloat16, device=meta),
        grid=(VIEW_GRID,) * views,
        input_ids=torch.zeros(1, length, dtype=torch.long, device=meta),
        attention_mask=torch.ones(1, length, dtype=torch.long, device=meta),
        position_ids=torch.zeros(3, 1, length, dtype=torch.long, device=meta),
        image_indices=torch.zeros(visual_tokens, dtype=torch.long, device=meta),
        state=torch.empty(1, 1, 132, dtype=torch.bfloat16, device=meta),
        embodiment_id=torch.zeros(1, dtype=torch.long, device=meta),
        noise=torch.empty(1, 40, 132, dtype=torch.bfloat16, device=meta), steps=2)
    assert outputs.vision_embeddings.shape == (visual_tokens, 2048)
    assert [tuple(feature.shape) for feature in outputs.deepstack] == [(visual_tokens, 2048)] * 3
    assert outputs.backbone_features.shape == (1, length, 2048)
    assert outputs.actions.shape == outputs.velocity_step_0.shape == (1, 40, 132)


def test_lingbot_official_schema_is_the_engine_layout() -> None:
    """LingBot's engine runs the official layout itself: the reference's
    schema is the frozen checkpoint's 1,555 tensors."""
    from flash_vla.models.lingbot import reference, spec

    schema = official_schema(reference.make_reference().parts(), prefixes=reference.PREFIXES)
    assert schema == dict(spec.WEIGHT_SHAPES)


def test_lingbot_reference_produces_every_stage_on_meta() -> None:
    from flash_vla.models.lingbot import reference

    meta = torch.device("meta")
    model = reference.load({name: torch.empty(shape, dtype=torch.bfloat16, device=meta)
                            for name, shape in official_schema(reference.make_reference().parts(),
                                                               prefixes=reference.PREFIXES).items()})
    assert model.vision.inverse_frequency.dtype == torch.bfloat16
    outputs = model(
        torch.empty(3, 256, 1176, dtype=torch.bfloat16, device=meta), grid=((1, 16, 16),) * 3,
        image_masks=torch.ones(3, dtype=torch.bool, device=meta),
        language_tokens=torch.zeros(1, 72, dtype=torch.long, device=meta),
        language_masks=torch.ones(1, 72, dtype=torch.bool, device=meta),
        state=torch.empty(1, 75, dtype=torch.bfloat16, device=meta),
        noise=torch.empty(1, 50, 75, dtype=torch.bfloat16, device=meta), steps=2, depth=2)
    assert outputs.vision_embeddings.shape == (3, 64, 2048)
    assert [tuple(key.shape) for key, _ in outputs.prefix.cache] == [(1, 264, 2, 128)] * 2
    assert outputs.actions.shape == outputs.velocity_step_0.shape == (1, 50, 75)
