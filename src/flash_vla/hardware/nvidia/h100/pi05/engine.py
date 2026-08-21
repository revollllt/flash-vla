"""`Pi05Inference`: weights, buffers, and three CUDA graphs for the forward pass.

Construction allocates every weight and buffer up front, precomputes the encoder
RoPE table, loads the checkpoint, warms the kernels, and captures three graphs.
`forward` then copies the inputs in, replays vision, tokenizes the state on the
host while that runs, copies the prompt inputs in, and replays prefix and
decoder.

Two of the splits earn their keep for different reasons.

**Vision from the rest** is a data dependency. Pi0 captures everything as one
graph because its prompt is fixed at load time and `forward` is pure GPU work.
Pi0.5's prompt carries the discretized state, so every call needs ~16 us of host
tokenization before the prefix can start -- strictly serial against a single
graph. Vision depends only on the images and takes roughly 2.4 ms, three orders
of magnitude more than the host work needs, so splitting there hides all of it.

**Prefix from decoder** is not a dependency, it is a measurement. Both could sit
in one graph; keeping them apart costs one extra launch and makes the per-stage
split readable by replaying one graph at a time, which is how the roofline in
PLAN.md §1.2 gets checked. It also gives the prefix parity gate an entry point
that does not run a decoder.

Capture constrains the design the same way it does in Pi0: nothing inside a pass
may allocate. Scratch comes from a `ScratchPool`, frozen after warmup so a
missed pre-allocation raises instead of silently allocating mid-capture.
"""
from __future__ import annotations

import torch

from flash_vla.models.pi05.spec import ENCODER_LAYERS, MAX_TOKEN_LEN, runtime_shapes
from flash_vla.runtime.cuda import ScratchPool

from . import pipeline
from .backends.tilelang import wrappers
from .buffers import allocate_static_buffers
from .ops import op_table
from .prefix import PrefixInputs


class Pi05Inference:
    """One captured Pi0.5 forward pass.

    `steps` and `layers` exist for bisection: shortening either keeps the
    pipeline intact while cutting depth, which is how parity is read -- on random
    weights a deep run diverges chaotically between any two implementations that
    are not bit-identical.
    """

    def __init__(self, checkpoint, tokenizer, num_views: int, chunk_size: int,
                 steps: int = 10, layers: int = ENCODER_LAYERS, fused: bool = True,
                 prompt_len: int = MAX_TOKEN_LEN, device: str = "cuda",
                 plan: dict[str, str] | None = None):
        """`plan` maps a call-site name to the backend that implements it.

        `None` keeps every call site on TileLang, which is the only backend in
        the tree today. A second backend registers in `backends/__init__.py` and
        is selected per call site here. Resolved before capture, so the replay
        path never dispatches -- the rule that makes a mixed backend free at
        runtime.
        """
        self.num_views = num_views
        self.chunk_size = chunk_size
        self.steps = steps
        self.layers = layers
        self.prompt_len = prompt_len
        self.plan = dict(plan) if plan else {}
        self.ops = op_table(fused, plan=self.plan or None)

        self.weights = {
            name: torch.empty(shape, dtype=torch.bfloat16, device=device)
            for name, shape in runtime_shapes(steps).items()
        }
        self.buffers, self.encoder_seq_len = allocate_static_buffers(
            num_views, chunk_size, device, prompt_len)

        missing = sorted(set(self.weights) - set(checkpoint))
        if missing:
            raise KeyError(f"checkpoint is missing {missing}; run models.pi05.weights.fold")
        for name, value in checkpoint.items():
            if name in self.weights:
                self.weights[name].copy_(value)

        self.inputs = PrefixInputs(tokenizer, num_views, chunk_size)
        self.pool = ScratchPool()
        self.vision_graph = torch.cuda.CUDAGraph()
        self.prefix_graph = torch.cuda.CUDAGraph()
        self.decoder_graph = torch.cuda.CUDAGraph()
        self._capture()

    # -- task ---------------------------------------------------------------

    def set_task(self, prompt: str) -> None:
        """Install the task string. Call whenever it changes; see `Pi05Tokenizer.set_task`."""
        self.inputs.tokenizer.set_task(prompt)

    # -- capture ------------------------------------------------------------

    def _run_vision(self):
        with wrappers.use_pool(self.pool):
            pipeline.vision(self.ops, self.weights, self.buffers, self.num_views)

    def _run_prefix(self):
        with wrappers.use_pool(self.pool):
            pipeline.prefix(self.ops, self.weights, self.buffers, self.num_views,
                            self.encoder_seq_len, layers=self.layers)

    def _run_decoder(self):
        with wrappers.use_pool(self.pool):
            pipeline.decoder(self.ops, self.weights, self.buffers, self.encoder_seq_len,
                             steps=self.steps, layers=self.layers)

    def _capture(self):
        """Warm up (compiling every kernel and filling the pool), freeze, then capture."""
        for _ in range(3):
            self._run_vision()
            self._run_prefix()
            self._run_decoder()
        torch.cuda.synchronize()
        self.pool.freeze()

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.vision_graph.capture_begin()
            self._run_vision()
            self.vision_graph.capture_end()
            self.prefix_graph.capture_begin()
            self._run_prefix()
            self.prefix_graph.capture_end()
            self.decoder_graph.capture_begin()
            self._run_decoder()
            self.decoder_graph.capture_end()
        torch.cuda.synchronize()

    # -- inference ----------------------------------------------------------

    def forward_prefix(self, images: torch.Tensor, state) -> int:
        """Run vision and the prefix; return the number of valid prefix rows.

        The KV cache is left in `buffers["encoder_K"]` / `["encoder_V"]`, rows
        `[:encoder_seq_len]`, of which `[:n_valid]` carry data and the rest are
        masked padding.

        Host tokenization sits between the two replays deliberately: the vision
        graph is already executing when it runs.
        """
        self.buffers["observation_images_normalized"].copy_(images)
        self.vision_graph.replay()

        n_valid = self.inputs.build(state)
        self.inputs.copy_into(self.buffers)

        self.prefix_graph.replay()
        return n_valid

    def forward(self, images: torch.Tensor, state, noise: torch.Tensor) -> torch.Tensor:
        """Copy the inputs in, replay the three graphs, return the denoised chunk.

        `noise` is copied up front rather than between the prefix and decoder
        replays, so nothing on the host separates them.
        """
        self.buffers["diffusion_noise"].copy_(noise)
        self.forward_prefix(images, state)
        self.decoder_graph.replay()
        return self.buffers["diffusion_noise"]

    @property
    def kv_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The prefix K and V the decoder attends over, trimmed to the prefix."""
        return (self.buffers["encoder_K"][:, :self.encoder_seq_len],
                self.buffers["encoder_V"][:, :self.encoder_seq_len])
