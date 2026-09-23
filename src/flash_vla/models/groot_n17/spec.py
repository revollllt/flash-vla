"""GR00T N1.7 LIBERO: the fixed workload's constants, without importing the model.

Values are the official `libero_10` checkpoint's (see README.md): two 256x256
views become 512 patches of width 1536 and 128 visual tokens; a 16-layer
Qwen3-VL backbone; a 32-block DiT action head denoising a 40x132 chunk in four
steps.
"""
from __future__ import annotations

MODEL_REVISION = "groot-n17-libero-v1"
INFERENCE_SIGNATURE = "groot-n17-qwen3vl16-dit32-libero-bf16-v1"

VIEWS = 2
IMAGE_SIZE = 256
#: Vision patches of all views, and their flattened width.
PATCHES = 512
PATCH_WIDTH = 1536
#: Patches per view (a 16x16 grid).
PATCHES_PER_VIEW = 256
VISUAL_TOKENS = 128
#: The prepared fixture's text sequence; a different length needs another fixture.
SEQUENCE_LENGTH = 156

VISION_BLOCKS = 24
VISION_DIM = 1024
VISION_FFN = 4096
#: Vision blocks whose output joins the first backbone layers (Qwen3-VL deepstack).
DEEPSTACK_LAYERS = 3

LAYERS = 16
BACKBONE_DIM = 2048
BACKBONE_FFN = 6144
KV_DIM = 1024

DIT_BLOCKS = 32
DIT_DIM = 1536
DIT_FFN = 6144

STATE_DIM = 132
ACTION_DIM = 132
CHUNK = 40
STEPS = 4


__all__ = ["ACTION_DIM", "BACKBONE_DIM", "BACKBONE_FFN", "CHUNK", "DEEPSTACK_LAYERS", "DIT_BLOCKS",
           "DIT_DIM", "DIT_FFN", "IMAGE_SIZE", "INFERENCE_SIGNATURE", "KV_DIM", "LAYERS",
           "MODEL_REVISION", "PATCHES", "PATCHES_PER_VIEW", "PATCH_WIDTH", "SEQUENCE_LENGTH",
           "STATE_DIM", "STEPS", "VIEWS", "VISION_BLOCKS", "VISION_DIM", "VISION_FFN",
           "VISUAL_TOKENS"]
