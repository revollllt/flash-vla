"""Three plain PyTorch stages, bound to the runner's buffers and weights."""
from __future__ import annotations

import torch

from flash_vla.models.groot_n17.reference import PREFIXES, bind_module, make_modules
from flash_vla.models.groot_n17.weights import weight_shapes
from flash_vla.runtime.ops import OpSpec

NAMES = ("groot_vision", "groot_backbone", "groot_action")
WEIGHTS = {part: tuple(name for name in weight_shapes() if name.startswith(prefix))
           for part, prefix in PREFIXES.items()}
PARAMS = {part: tuple(f"w{i}" for i in range(len(names))) for part, names in WEIGHTS.items()}


def _vision_flops(s):
    rows, patch = s["pixels"]
    dim, ffn = 1024, 4096
    blocks = 24 * (8 * rows * dim * dim + 4 * rows * dim * ffn + 4 * 2 * 256**2 * dim)
    mergers = 4 * 2 * 128 * (4096**2 + 4096 * 2048)
    return 2 * rows * patch * dim + blocks + mergers


def _backbone_flops(s):
    rows = s["input_ids"][1]
    return 16 * (2 * rows * 2048 * (2048 + 1024 + 1024 + 2048)
                 + 6 * rows * 2048 * 6144 + 4 * rows * rows * 2048)


def _action_flops(s):
    """Dense matmul/attention FLOPs for this Target's four-step workload; excludes pointwise ops."""
    prefix, rows, dim = s["backbone"][1], 41, 1536
    refiner = 4 * (8 * prefix * 2048**2 + 16 * prefix * 2048**2 + 4 * prefix**2 * 2048)
    cross = 4 * rows * dim**2 + 4 * prefix * 2048 * dim + 4 * rows * prefix * dim
    self_attn = 8 * rows * dim**2 + 4 * rows**2 * dim
    ffn = 4 * rows * dim * 6144
    modulation = 2 * dim * (2 * dim)
    dit = 16 * (cross + self_attn) + 32 * (ffn + modulation)
    encoder = 2 * 40 * (132 * dim + 2 * dim**2 + dim**2)
    decoder = 2 * rows * (1024**2 + 1024 * 132)
    timestep = 2 * (256 * dim + dim**2)
    output = 2 * (dim * 2 * dim + rows * dim * 1024)
    state = 2 * (132 * 1024 + 1024 * dim)
    return refiner + state + 4 * (dit + encoder + decoder + timestep + output)


OPS = (
    OpSpec("groot_vision", ("pixels", "out", "deepstack") + PARAMS["vision"],
           outputs=("out", "deepstack"), weights=PARAMS["vision"], flops=_vision_flops),
    OpSpec("groot_backbone", ("input_ids", "attention_mask", "position_ids", "image_indices",
           "vision", "deepstack", "out") + PARAMS["backbone"], outputs=("out",),
           weights=PARAMS["backbone"], flops=_backbone_flops),
    OpSpec("groot_action", ("backbone", "state", "noise", "embodiment", "input_ids",
           "attention_mask", "out", "velocity") + PARAMS["action"], outputs=("out", "velocity"),
           weights=PARAMS["action"], flops=_action_flops),
)


def make_wrappers(scratch, selected_names=None):
    modules = {}
    vision_tables = {}
    grid = torch.tensor([[1, 16, 16], [1, 16, 16]], device="cpu")
    cu_seqlens = torch.tensor([0, 256, 512], dtype=torch.int32, device="cpu")

    def module(part, tensors):
        if part not in modules:
            empty = make_modules()[part]
            modules[part] = bind_module(empty, dict(zip(WEIGHTS[part], tensors)),
                                        prefix=PREFIXES[part], device=tensors[0].device)
        return modules[part]

    @torch.no_grad()
    def vision(pixels, out, deepstack, *weights):
        visual = module("vision", weights)
        if not vision_tables:
            # Upstream uploads Python-built indices inside forward, which CUDA
            # Graph capture forbids. Prepare these fixed-grid tables during warmup.
            position = visual.fast_pos_embed_interpolate(grid)
            rope = visual.rot_pos_emb(grid)
            rope = torch.cat((rope, rope), dim=-1)
            for name, value in (("position", position), ("cos", rope.cos()), ("sin", rope.sin())):
                table = scratch(f"groot_vision_{name}", value.shape, value.dtype, value.device)
                table.copy_(value)
                vision_tables[name] = table
        hidden = visual.patch_embed(pixels) + vision_tables["position"]
        for index, block in enumerate(visual.blocks):
            hidden = block(hidden, cu_seqlens=cu_seqlens,
                           position_embeddings=(vision_tables["cos"], vision_tables["sin"]))
            if index in visual.deepstack_visual_indexes:
                slot = visual.deepstack_visual_indexes.index(index)
                deepstack[slot].copy_(visual.deepstack_merger_list[slot](hidden))
        out.copy_(visual.merger(hidden))

    @torch.no_grad()
    def backbone(input_ids, attention_mask, position_ids, image_indices, vision,
                 deepstack, out, *weights):
        language = module("backbone", weights)
        hidden = language.embed_tokens(input_ids)
        hidden.index_copy_(1, image_indices, vision.unsqueeze(0))
        length = input_ids.shape[1]
        allowed = torch.ones((length, length), dtype=torch.bool, device=hidden.device).tril()
        allowed = allowed[None, None] & attention_mask[:, None, None, :].bool()
        positions = torch.arange(length, device=hidden.device)
        rotary = language.rotary_emb(hidden, position_ids)
        for index, layer in enumerate(language.layers):
            hidden = layer(hidden, attention_mask=allowed, position_ids=position_ids[0],
                           cache_position=positions, position_embeddings=rotary, use_cache=False)
            if index < 3:
                # Fixed-size gather/scatter avoids Qwen's dynamic CUDA nonzero during capture.
                values = hidden.index_select(1, image_indices) + deepstack[index].unsqueeze(0)
                hidden.index_copy_(1, image_indices, values)
        # GR00T reads ForConditionalGeneration.hidden_states[-1]. In the
        # supported Transformers version this is the final decoder output,
        # before the language model's final RMSNorm.
        out.copy_(hidden)

    @torch.no_grad()
    def action(backbone, state, noise, embodiment, input_ids, attention_mask, out, velocity, *weights):
        head = module("action", weights)
        actions, first_velocity = head(backbone, state, noise, embodiment,
                                       input_ids == 151655, attention_mask.bool(), steps=4)
        out.copy_(actions)
        velocity.copy_(first_velocity)

    wrappers = dict(zip(NAMES, (vision, backbone, action)))
    return {name: wrappers[name] for name in (selected_names or NAMES)}
