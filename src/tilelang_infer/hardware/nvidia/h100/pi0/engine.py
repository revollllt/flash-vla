"""`Pi0Inference`: weights, buffers, and one CUDA graph for the whole forward pass.

Construction allocates every weight and buffer up front, precomputes the RoPE
tables, loads the checkpoint, warms the kernels, and captures a single graph
covering vision, encoder and decoder. `forward` then copies the three inputs
into their static buffers and replays.

Capture is what makes the numbers reproducible, and it constrains the design:
nothing inside the pass may allocate. Scratch that the wrappers need comes from
a `ScratchPool`, which is frozen after warmup so a missed pre-allocation raises
instead of silently allocating mid-capture.
"""
from __future__ import annotations

import torch

from tilelang_infer.models.pi0.spec import weight_shapes
from tilelang_infer.runtime.cuda import ScratchPool

from . import pipeline, wrappers
from .buffers import allocate_static_buffers
from .ops import op_table


class Pi0Inference:
    """One captured Pi0 forward pass.

    steps and layers exist for bisection: shortening either keeps the pipeline
    intact while cutting depth, which is how parity is read -- on random weights
    a deep run diverges chaotically between any two implementations that are not
    bit-identical.
    """

    def __init__(self, checkpoint, num_views: int, chunk_size: int, steps: int = 10,
                 layers: int = 18, fused: bool = True, device: str = "cuda"):
        self.num_views = num_views
        self.chunk_size = chunk_size
        self.steps = steps
        self.layers = layers
        self.fused = fused
        self.ops = op_table(fused)
        self.prompt_len = len(checkpoint["language_embeds"])

        bf16 = torch.bfloat16
        self.weights = {name: torch.empty(shape, dtype=bf16, device=device)
                        for name, shape in weight_shapes(self.prompt_len).items()}
        self.buffers, self.encoder_seq_len = allocate_static_buffers(
            num_views, chunk_size, self.prompt_len, device)

        for name, value in checkpoint.items():
            self.weights[name].copy_(value)

        self.pool = ScratchPool()
        self.graph = torch.cuda.CUDAGraph()
        self._capture()

    def _run(self):
        """One full forward pass, in place on the static buffers."""
        self.buffers["encoder_x"][self.num_views * 256:].copy_(self.weights["language_embeds"])
        with wrappers.use_pool(self.pool):
            pipeline.vision_encoder(self.ops, self.weights, self.buffers, self.num_views)
            pipeline.transformer_encoder(self.ops, self.weights, self.buffers, self.encoder_seq_len)
            pipeline.transformer_decoder(self.ops, self.weights, self.buffers,
                                         self.encoder_seq_len, steps=self.steps, layers=self.layers)

    def _capture(self):
        """Warm up (compiling every kernel and filling the pool), freeze, then capture."""
        for _ in range(3):
            self._run()
        torch.cuda.synchronize()
        self.pool.freeze()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.graph.capture_begin()
            self._run()
            self.graph.capture_end()
        torch.cuda.synchronize()

    def forward(self, images, state, noise):
        """Copy inputs into the static buffers, replay the graph, return the denoised chunk."""
        self.buffers["observation_images_normalized"].copy_(images)
        self.buffers["observation_state_normalized"].copy_(state)
        self.buffers["diffusion_noise"].copy_(noise)
        self.graph.replay()
        return self.buffers["diffusion_noise"]
