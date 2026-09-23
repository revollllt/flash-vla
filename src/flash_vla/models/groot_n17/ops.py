"""The GR00T graph's call sites: vision, backbone and action head as one op each.

Each stage takes every weight of its module as a numbered parameter
(`PARAMS`), bound to the checkpoint names of `WEIGHTS`. The FLOP formulas
count dense matmuls and attention of the fixed workload, excluding pointwise
ops; they are what the floor model prices.
"""
from __future__ import annotations

from flash_vla.runtime.ops import OpSpec, Shapes

from .reference import PREFIXES
from .spec import (
    ACTION_DIM,
    BACKBONE_DIM,
    BACKBONE_FFN,
    CHUNK,
    DIT_BLOCKS,
    DIT_DIM,
    DIT_FFN,
    KV_DIM,
    LAYERS,
    PATCHES_PER_VIEW,
    STATE_DIM,
    STEPS,
    VIEWS,
    VISION_BLOCKS,
    VISION_DIM,
    VISION_FFN,
    VISUAL_TOKENS,
)
from .weights import weight_shapes

#: Hidden width of the embodiment-conditioned state MLP and action decoder.
HEAD_HIDDEN = 1024
#: Sinusoidal channels of the flow-matching timestep embedding.
TIMESTEP_CHANNELS = 256

#: The three monolithic call sites of the GR00T graph.
CALL_SITES = ("groot_vision", "groot_backbone", "groot_action")
#: Checkpoint weight names of each stage, in the order its op takes them.
WEIGHTS = {part: tuple(name for name in weight_shapes() if name.startswith(prefix))
           for part, prefix in PREFIXES.items()}
#: The positional weight parameter names of each stage's op.
PARAMS = {part: tuple(f"w{i}" for i in range(len(names))) for part, names in WEIGHTS.items()}


def vision_flops(shapes: Shapes) -> int:
    rows, patch = shapes["pixels"]
    dim, ffn = VISION_DIM, VISION_FFN
    blocks = VISION_BLOCKS * (8 * rows * dim * dim + 4 * rows * dim * ffn
                              + 4 * VIEWS * PATCHES_PER_VIEW**2 * dim)
    merged = 4 * VISION_DIM
    mergers = 4 * 2 * VISUAL_TOKENS * (merged**2 + merged * BACKBONE_DIM)
    return 2 * rows * patch * dim + blocks + mergers


def backbone_flops(shapes: Shapes) -> int:
    rows = shapes["input_ids"][1]
    dim = BACKBONE_DIM
    return LAYERS * (2 * rows * dim * (dim + KV_DIM + KV_DIM + dim)
                     + 6 * rows * dim * BACKBONE_FFN + 4 * rows * rows * dim)


def action_flops(shapes: Shapes) -> int:
    """Dense matmul/attention FLOPs of the four-step workload; excludes pointwise ops."""
    prefix, rows, dim = shapes["backbone"][1], CHUNK + 1, DIT_DIM
    backbone = BACKBONE_DIM
    refiner = 4 * (8 * prefix * backbone**2 + 16 * prefix * backbone**2 + 4 * prefix**2 * backbone)
    cross = 4 * rows * dim**2 + 4 * prefix * backbone * dim + 4 * rows * prefix * dim
    self_attn = 8 * rows * dim**2 + 4 * rows**2 * dim
    ffn = 4 * rows * dim * DIT_FFN
    modulation = 2 * dim * (2 * dim)
    # Blocks alternate cross- and self-attention; every block has an FFN.
    dit = DIT_BLOCKS // 2 * (cross + self_attn) + DIT_BLOCKS * (ffn + modulation)
    encoder = 2 * CHUNK * (ACTION_DIM * dim + 2 * dim**2 + dim**2)
    decoder = 2 * rows * (HEAD_HIDDEN**2 + HEAD_HIDDEN * ACTION_DIM)
    timestep = 2 * (TIMESTEP_CHANNELS * dim + dim**2)
    output = 2 * (dim * 2 * dim + rows * dim * HEAD_HIDDEN)
    state = 2 * (STATE_DIM * HEAD_HIDDEN + HEAD_HIDDEN * dim)
    return refiner + state + STEPS * (dit + encoder + decoder + timestep + output)


OPS = (
    OpSpec("groot_vision", ("pixels", "out", "deepstack") + PARAMS["vision"],
           outputs=("out", "deepstack"), weights=PARAMS["vision"], flops=vision_flops),
    OpSpec("groot_backbone", ("input_ids", "attention_mask", "position_ids", "image_indices",
           "vision", "deepstack", "out") + PARAMS["backbone"], outputs=("out",),
           weights=PARAMS["backbone"], flops=backbone_flops),
    OpSpec("groot_action", ("backbone", "state", "noise", "embodiment", "input_ids",
           "attention_mask", "out", "velocity") + PARAMS["action"], outputs=("out", "velocity"),
           weights=PARAMS["action"], flops=action_flops),
)

__all__ = ["CALL_SITES", "OPS", "PARAMS", "WEIGHTS"]
