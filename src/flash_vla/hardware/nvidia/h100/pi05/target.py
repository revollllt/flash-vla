"""The H100 / Pi0.5 Target: the model contract and its computation graph.

`Pi05` declares what the runner needs and nothing that runs: the identity
axes, the configuration, the shape numbers, the weight schema and loader, the
backend registry with the one shipped plan and the reference plan, the host
slot that tokenizes the state, and the graph (`pipeline.build`). The runner
(`flash_vla.runtime.ModelRunner`) does the rest.

`steps` and `layers` exist for bisection: shortening either keeps the graph
intact while cutting depth, which is how a reference check is read -- on
random weights a deep run diverges chaotically between any two
implementations that are not bit-identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from flash_vla.models.pi05.spec import (
    ACTION_DIM,
    DECODER_DIM,
    DECODER_FFN,
    DECODER_HEADS,
    ENCODER_DIM,
    ENCODER_FFN,
    ENCODER_LAYERS,
    HEAD_DIM,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    KV_HEADS,
    MAX_TOKEN_LEN,
    MODEL_REVISION,
    QKV_WIDTH,
    STATE_DIM,
    VISION_DIM,
    VISION_FFN,
    VISION_HEAD_DIM,
    VISION_HEADS,
    VISION_LAYERS,
    VISION_TOKENS,
    runtime_shapes,
)
from flash_vla.runtime import VLA, Input
from flash_vla.runtime.cost import Ceiling
from flash_vla.runtime.graph import Graph

from . import pipeline
from .backends import REGISTRY
from .prefix import PrefixInputs


@dataclass(frozen=True)
class Pi05Config:
    """Construction inputs beyond the checkpoint and the plan."""
    num_views: int = 3
    chunk_size: int = 50
    steps: int = 10
    layers: int = ENCODER_LAYERS
    prompt_len: int = MAX_TOKEN_LEN
    #: A `Pi05Tokenizer`, required to run; `None` only declares the graph.
    tokenizer: Any = None
    #: The task string installed at construction; `set_task` changes it later.
    prompt: str | None = None


class Pi05(VLA):
    """Pi0.5 on H100 SXM5: three stages, one host slot, bf16."""

    name = "hardware/nvidia/h100/pi05"
    hardware = "h100-sxm5-80gb"
    model = "pi05"
    model_revision = MODEL_REVISION
    precision = "bf16"

    INPUTS = (
        Input("images", lambda s: (s["num_views"], IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS),
              torch.bfloat16, "images"),
        # The state is tokenized on the host into the prompt buffers.
        Input("state", lambda s: (STATE_DIM,), torch.float32, None),
        Input("noise", lambda s: (s["chunk"], ACTION_DIM), torch.bfloat16, "actions"),
    )

    registry = REGISTRY
    #: The shipped plan: the fused backbone attention, now from the shared
    #: Gemma component package rather than this Target's own copy of the same
    #: kernel source (proven bit-identical, job 599788), and the decoder's
    #: attention and FFN halves on the CUDA backend with the PDL chain armed.
    plan = {
        "llm_backbone_attention": "gemma-cuda",
        "action_expert_norm_qkv_rope": "cuda-pdl",
        "action_expert_attention": "cuda-pdl",
        "action_expert_out_proj_residual": "cuda-pdl",
        "action_expert_norm_gated_ffn": "cuda-pdl",
        "action_expert_ffn_down_residual": "cuda-pdl",
    }
    #: The reference route: every call site on TileLang, the backbone
    #: attention on the torch chain.
    reference_plan: Mapping[str, str] = {}
    #: The gate/up GEMM's 16.8 MB of weights delivered cold by the copy engine
    #: at this call site's own geometry (tma_ring sweep Q, one node, one job):
    #: the machine's number for the phase, used in place of the constants' rule.
    CEILINGS = {"action_expert_norm_gated_ffn": Ceiling(us=9.41, tag="tma.bw.dev.burst", job=591174)}

    def configure(self, **config: Any) -> Pi05Config:
        return Pi05Config(**config)

    def shape(self, config: Pi05Config, checkpoint=None) -> dict[str, int]:
        visual_tokens = config.num_views * VISION_TOKENS
        return {
            "batch": 1,
            "num_views": config.num_views,
            "image_height": IMAGE_SIZE,
            "image_width": IMAGE_SIZE,
            "image_channels": IMAGE_CHANNELS,
            "visual_tokens_per_view": VISION_TOKENS,
            "visual_tokens": visual_tokens,
            "vision_dim": VISION_DIM,
            "vision_ffn_dim": VISION_FFN,
            "vision_heads": VISION_HEADS,
            "vision_head_dim": VISION_HEAD_DIM,
            "vision_layers": VISION_LAYERS,
            "prompt_len": config.prompt_len,
            "prefix_len": visual_tokens + config.prompt_len,
            "chunk": config.chunk_size,
            "expert_tokens": config.chunk_size,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "steps": config.steps,
            "layers": config.layers,
            "encoder_dim": ENCODER_DIM,
            "encoder_ffn_dim": ENCODER_FFN,
            "query_heads": DECODER_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "qkv_width": QKV_WIDTH,
            "expert_dim": DECODER_DIM,
            "expert_ffn_dim": DECODER_FFN,
        }

    def weight_shapes(self, shape: Mapping[str, int]) -> Mapping[str, tuple[int, ...]]:
        return runtime_shapes(shape["steps"])

    def load_weights(self, checkpoint, weights) -> None:
        missing = sorted(set(weights) - set(checkpoint))
        if missing:
            raise KeyError(f"checkpoint is missing {missing}; run models.pi05.weights.fold")
        for name, value in checkpoint.items():
            if name in weights:
                weights[name].copy_(value)

    def build(self, g: Graph, shape: Mapping[str, int]) -> None:
        pipeline.build(g, shape)

    def host_state(self, config: Pi05Config, shape: Mapping[str, int]) -> PrefixInputs:
        if config.tokenizer is None:
            raise ValueError("running Pi0.5 needs a tokenizer; pass tokenizer=Pi05Tokenizer(...)")
        if config.prompt is not None:
            config.tokenizer.set_task(config.prompt)
        return PrefixInputs(config.tokenizer, config.num_views, config.chunk_size)

    def host(self, slot: str, *, host_state: PrefixInputs, buffers, state=None, **_) -> None:
        """Tokenize `state` and stage the prompt inputs; sits after vision starts."""
        if slot != "prompt":
            raise KeyError(f"no host slot {slot!r}; this Target has 'prompt'")
        host_state.build(state)
        host_state.copy_into(buffers)


TARGET = Pi05()


def set_task(runner, prompt: str) -> None:
    """Install the task string. Call whenever it changes; see `Pi05Tokenizer.set_task`."""
    runner.host_state.tokenizer.set_task(prompt)


def forward_prefix(runner, images: torch.Tensor, state) -> int:
    """Run the vision encoder, the host slot and the backbone; return the valid prefix rows.

    The KV cache is left in `buffers["prefix_k"]` / `["prefix_v"]`, of which
    the first `n_valid` rows carry data and the rest are masked padding.
    """
    runner.buffers["images"].copy_(images)
    runner.replay("vision_encoder")
    runner.host("prompt", state=state)
    runner.replay("llm_backbone")
    return runner.host_state.n_valid


__all__ = ["Pi05", "Pi05Config", "TARGET", "forward_prefix", "set_task"]
