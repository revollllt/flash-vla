"""LingBot-VLA as a model definition: identity, configuration, shapes and inputs.

The model runs at one frozen shape. Its inputs are a recorded fixture (the
`fixture` asset), which the upstream processor and tokenizer produced; a
synthetic construction (`LingBotConfig.synthetic`), which has no fixture,
draws them from the seed instead (`sources.runner_source`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from safetensors.torch import load_file
import torch

from flash_vla.runtime.graph import Graph
from flash_vla.runtime.vla import (CheckpointReader, ConfigValue, Input, ModelDefinition,
                                   ReplayAxis, Workload)

from . import graph, work
from .ops import OPS
from .spec import (
    ACTION_DIM,
    BACKBONE_DIM,
    BACKBONE_FFN,
    CHUNK,
    EXPERT_DIM,
    EXPERT_FFN,
    FLOW_STEPS,
    HEAD_DIM,
    IMAGE_SIZE,
    INFERENCE_SIGNATURE,
    KV_HEADS,
    LANGUAGE_SLOTS,
    LAYERS,
    MODEL_REVISION,
    PATCH_ROWS_PER_VIEW,
    PATCH_WIDTH,
    PREFIX_LEN,
    QUERY_HEADS,
    STATE_DIM,
    SUFFIX_LEN,
    VIEWS,
    VISION_DIM,
    VISION_FFN,
    VISION_HEAD_DIM,
    VISION_HEADS,
    VISION_LAYERS,
    VISUAL_TOKENS_PER_VIEW,
    VOCABULARY,
    WEIGHT_SHAPES,
)

#: The seed the frozen fixture was recorded with; it is the only one it provides.
FIXTURE_SEED = 42
#: Valid prompt tokens of the synthetic inputs; the rest of the slots are padding.
SYNTHETIC_PROMPT_TOKENS = 48


@dataclass(frozen=True)
class LingBotConfig:
    steps: int = FLOW_STEPS
    layers: int = LAYERS
    #: Inputs drawn from the seed rather than read from the recorded fixture.
    synthetic: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.steps <= FLOW_STEPS:
            raise ValueError(f"steps must be in [1, {FLOW_STEPS}]")
        if not 1 <= self.layers <= LAYERS:
            raise ValueError(f"layers must be in [1, {LAYERS}]")


class LingBotModel(ModelDefinition[LingBotConfig, None]):
    """LingBot-VLA: vision, prefix and action stages, one monolithic call site each."""

    name = "lingbot-vla"
    model_revision = MODEL_REVISION
    inference_signature = INFERENCE_SIGNATURE
    shape_axes = (
        "batch", "views", "image_height", "image_width", "patch_rows_per_view",
        "patch_width", "visual_tokens_per_view", "visual_tokens", "vision_layers",
        "vision_dim", "vision_ffn_dim", "vision_heads", "vision_head_dim",
        "language_slots", "prefix_len", "state_dim", "action_dim", "chunk",
        "suffix_len", "steps", "layers", "backbone_dim", "backbone_ffn_dim",
        "expert_dim", "expert_ffn_dim", "query_heads", "kv_heads", "head_dim",
    )
    #: The valid language tokens after the image tokens: the prefix masks the
    #: rest, and every stage runs all 72 slots (no Target buckets them).
    replay_axis = ReplayAxis(name="valid_language_tokens", limit="language_slots",
                             offset="visual_tokens", slot=None,
                             stages=("llm_backbone", "action_expert"),
                             extent=lambda host_state, inputs: int(inputs["language_masks"].sum()))
    workloads = (
        # RoboDojo: XPolicyLab `LingBot_VLA` (`train_multinode_robodojo.sh`): three
        # 224x224 cameras, 72 language slots, chunk 50 -- the shapes `spec` fixes,
        # so the workload names no option (docs/workloads.md).
        Workload("robodojo", {}),
    )
    inputs = (
        Input("pixel_values", lambda s: (VIEWS, PATCH_ROWS_PER_VIEW, PATCH_WIDTH),
              torch.bfloat16, "pixel_values"),
        Input("image_masks", lambda s: (VIEWS,), torch.bool, "image_masks"),
        Input("language_tokens", lambda s: (1, LANGUAGE_SLOTS), torch.int64,
              "language_tokens"),
        Input("language_masks", lambda s: (1, LANGUAGE_SLOTS), torch.bool,
              "language_masks"),
        Input("state", lambda s: (1, STATE_DIM), torch.bfloat16, "state"),
        Input("noise", lambda s: (1, CHUNK, ACTION_DIM), torch.bfloat16, "noise"),
    )
    stage_outputs = {
        "vision_encoder": (("vision_embeddings", None),),
        "llm_backbone": (("prefix_k", 0), ("prefix_v", 0)),
        "action_expert": (("velocity_step_0", None), ("actions", None)),
    }
    ops = OPS
    work_rules = work.RULES
    reference_run = staticmethod(work.reference_run)

    def configure(self, **config: ConfigValue) -> LingBotConfig:
        return LingBotConfig(**config)

    def shape(self, config: LingBotConfig, checkpoint: CheckpointReader | None) -> dict[str, int]:
        return {
            "batch": 1,
            "views": VIEWS,
            "image_height": IMAGE_SIZE,
            "image_width": IMAGE_SIZE,
            "patch_rows_per_view": PATCH_ROWS_PER_VIEW,
            "patch_width": PATCH_WIDTH,
            "visual_tokens_per_view": VISUAL_TOKENS_PER_VIEW,
            "visual_tokens": VIEWS * VISUAL_TOKENS_PER_VIEW,
            "vision_layers": VISION_LAYERS,
            "vision_dim": VISION_DIM,
            "vision_ffn_dim": VISION_FFN,
            "vision_heads": VISION_HEADS,
            "vision_head_dim": VISION_HEAD_DIM,
            "language_slots": LANGUAGE_SLOTS,
            "prefix_len": PREFIX_LEN,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "chunk": CHUNK,
            "suffix_len": SUFFIX_LEN,
            "steps": config.steps,
            "layers": config.layers,
            "backbone_dim": BACKBONE_DIM,
            "backbone_ffn_dim": BACKBONE_FFN,
            "expert_dim": EXPERT_DIM,
            "expert_ffn_dim": EXPERT_FFN,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
        }

    def weight_shapes(self, shape: Mapping[str, int]) -> Mapping[str, tuple[int, ...]]:
        return WEIGHT_SHAPES

    def build(self, g: Graph, shape: Mapping[str, int]) -> None:
        graph.build(g, shape)

    def sample_inputs(self, config: LingBotConfig, shape: Mapping[str, int], seed: int,
                      device: torch.device, assets: Mapping[str, Path]) -> dict[str, torch.Tensor]:
        """The recorded fixture's inputs, which it provides only for
        `FIXTURE_SEED`; a synthetic construction's are drawn from `seed`: unit
        normal patches, state and noise, every view valid but the last, and
        `SYNTHETIC_PROMPT_TOKENS` random prompt tokens before padding."""
        if config.synthetic:
            generator = torch.Generator(device=device).manual_seed(seed)
            return {
                "pixel_values": torch.randn((VIEWS, PATCH_ROWS_PER_VIEW, PATCH_WIDTH), generator=generator,
                                            device=device).to(torch.bfloat16),
                "image_masks": torch.arange(VIEWS, device=device) < VIEWS - 1,
                "language_tokens": torch.randint(0, VOCABULARY, (1, LANGUAGE_SLOTS), generator=generator,
                                                 device=device),
                "language_masks": (torch.arange(LANGUAGE_SLOTS, device=device)
                                   < SYNTHETIC_PROMPT_TOKENS)[None],
                "state": torch.randn((1, STATE_DIM), generator=generator, device=device).to(torch.bfloat16),
                "noise": torch.randn((1, CHUNK, ACTION_DIM), generator=generator,
                                     device=device).to(torch.bfloat16),
            }
        if seed != FIXTURE_SEED:
            raise ValueError(f"the frozen LingBot fixture uses seed {FIXTURE_SEED}")
        values = load_file(Path(assets["fixture"]))
        return {spec.name: values[spec.name].to(device=device) for spec in self.inputs}


__all__ = ["FIXTURE_SEED", "LingBotConfig", "LingBotModel", "SYNTHETIC_PROMPT_TOKENS"]
