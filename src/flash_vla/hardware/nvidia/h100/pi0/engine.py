"""`Pi0Inference`: weights, buffers, and one graph segment for the whole forward pass.

Construction binds the plan, materializes the buffer plan, precomputes the RoPE
tables, loads the checkpoint, and runs the runtime lifecycle -- warm up, freeze
the scratch pool, capture a single segment covering vision, encoder and
decoder. `forward` then copies the three inputs into their static buffers and
replays.

Capture is what makes the numbers reproducible, and it constrains the design:
nothing inside the pass may allocate. Scratch that the wrappers need comes from
a `ScratchPool`, which is frozen after warmup so a missed pre-allocation raises
instead of silently allocating mid-capture.
"""
from __future__ import annotations

import torch

from flash_vla.models.pi0.spec import weight_shapes
from flash_vla.runtime import Identity
from flash_vla.runtime.cuda import Program, ScratchPool, Segment, StaticArena, Step

from . import pipeline
from .backends.tilelang import wrappers
from .buffers import buffer_plan
from .ops import op_table, resolve_plan

TARGET = "hardware/nvidia/h100/pi0"
HARDWARE = "h100-sxm5-80gb"
MODEL = "pi0"
PRECISION = "bf16"


class Pi0Inference:
    """One captured Pi0 forward pass.

    steps and layers exist for bisection: shortening either keeps the pipeline
    intact while cutting depth, which is how parity is read -- on random weights
    a deep run diverges chaotically between any two implementations that are not
    bit-identical.
    """

    #: One segment and no host work: the engine protocol's `program`.
    program: tuple[Step, ...] = (Step("segment", "forward"),)
    #: What the segment produces: the chunk and the layer-major KV cache.
    stage_outputs = {
        "forward": (("diffusion_noise", None), ("encoder_K", 0), ("encoder_V", 0)),
    }

    def __init__(self, checkpoint, num_views: int, chunk_size: int, steps: int = 10,
                 layers: int = 18, fused: bool = True, device: str = "cuda"):
        self.num_views = num_views
        self.chunk_size = chunk_size
        self.steps = steps
        self.layers = layers
        self.fused = fused
        self.device = torch.device(device)
        self.ops = op_table(fused)
        self.prompt_len = len(checkpoint["language_embeds"])
        self.identity = Identity(
            target=TARGET, hardware=HARDWARE, model=MODEL,
            shape={"num_views": num_views, "chunk": chunk_size, "steps": steps,
                   "layers": layers, "prompt_len": self.prompt_len},
            plan=resolve_plan(None), precision=PRECISION,
            options={"fused": fused})

        bf16 = torch.bfloat16
        self.weights = {name: torch.empty(shape, dtype=bf16, device=device)
                        for name, shape in weight_shapes(self.prompt_len).items()}
        plan_spec, self.encoder_seq_len = buffer_plan(num_views, chunk_size, self.prompt_len)
        self.arena = StaticArena(plan_spec, device)
        self.buffers = self.arena.buffers

        for name, value in checkpoint.items():
            self.weights[name].copy_(value)

        self.pool = ScratchPool()
        self.graphs = Program([Segment("forward", self._run)], self.pool)

    def _run(self):
        """One full forward pass, in place on the static buffers."""
        self.buffers["encoder_x"][self.num_views * 256:].copy_(self.weights["language_embeds"])
        with wrappers.use_pool(self.pool):
            pipeline.vision_encoder(self.ops, self.weights, self.buffers, self.num_views)
            pipeline.transformer_encoder(self.ops, self.weights, self.buffers, self.encoder_seq_len)
            pipeline.transformer_decoder(self.ops, self.weights, self.buffers,
                                         self.encoder_seq_len, steps=self.steps, layers=self.layers)

    def replay(self, segment: str = "forward") -> None:
        """Replay the captured segment on the current stream."""
        self.graphs.replay(segment)

    def host(self, slot: str, **_) -> None:
        """Pi0 has no host slot; every input is a device copy."""
        raise KeyError(f"no host slot {slot!r}; this engine has none")

    def allocation(self, name: str) -> torch.Tensor:
        """The base allocation behind buffer `name`, padding included."""
        return self.arena.allocation(name)

    def stage(self, images, state, noise, **_) -> None:
        """Copy the three inputs into their static buffers."""
        self.buffers["observation_images_normalized"].copy_(images)
        self.buffers["observation_state_normalized"].copy_(state)
        self.buffers["diffusion_noise"].copy_(noise)

    def sample_inputs(self, seed: int = 0) -> dict[str, torch.Tensor]:
        """Seeded random inputs at this engine's shapes: images, state, noise."""
        generator = torch.Generator(device=self.device).manual_seed(seed)

        def randn(*shape):
            return torch.randn(shape, generator=generator, device=self.device,
                               dtype=torch.bfloat16)

        return {"images": randn(self.num_views, 224, 224, 3), "state": randn(32),
                "noise": randn(self.chunk_size, 32)}

    def forward(self, images, state, noise):
        """Copy inputs into the static buffers, replay the segment, return the denoised chunk."""
        self.stage(images=images, state=state, noise=noise)
        self.replay("forward")
        return self.buffers["diffusion_noise"]
