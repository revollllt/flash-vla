"""`Pi05Prefix`: weights, buffers, and two CUDA graphs for the prefix pass.

Construction allocates every weight and buffer up front, precomputes the encoder
RoPE table, loads the checkpoint, warms the kernels, and captures two graphs.
`forward` then copies the images in, replays the vision graph, tokenizes the
state on the host while that runs, copies the prompt inputs in, and replays the
prefix graph.

The split is the point. Pi0 captures everything as one graph because its prompt
is fixed at load time and `forward` is pure GPU work. Pi0.5's prompt carries the
discretized state, so every call needs ~16 us of host tokenization before the
prefix can start -- strictly serial against a single graph. Vision depends only
on the images and takes roughly 2.4 ms, which is three orders of magnitude more
than the host work needs, so splitting there hides all of it. The cost is one
extra graph launch and the loss of kernel overlap across the boundary.

Capture constrains the design the same way it does in Pi0: nothing inside a pass
may allocate. Scratch comes from a `ScratchPool`, frozen after warmup so a
missed pre-allocation raises instead of silently allocating mid-capture.

This target covers the prefix only. The decoder's AdaRMSNorm call sites are
blocked on the tile-dataflow spec (PLAN.md §2.4, §4.3).
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


class Pi05Prefix:
    """One captured Pi0.5 prefix pass: images and state in, KV cache out.

    `layers` exists for bisection: shortening it keeps the pipeline intact while
    cutting depth, which is how parity is read -- on random weights a deep run
    diverges chaotically between any two implementations that are not
    bit-identical.
    """

    def __init__(self, checkpoint, tokenizer, num_views: int, chunk_size: int,
                 steps: int = 10, layers: int = ENCODER_LAYERS, fused: bool = True,
                 prompt_len: int = MAX_TOKEN_LEN, device: str = "cuda"):
        self.num_views = num_views
        self.chunk_size = chunk_size
        self.layers = layers
        self.prompt_len = prompt_len
        self.ops = op_table(fused)

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

    def _capture(self):
        """Warm up (compiling every kernel and filling the pool), freeze, then capture."""
        for _ in range(3):
            self._run_vision()
            self._run_prefix()
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
        torch.cuda.synchronize()

    # -- inference ----------------------------------------------------------

    def forward(self, images: torch.Tensor, state) -> int:
        """Run the prefix pass; return the number of valid prefix rows.

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

    @property
    def kv_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The prefix K and V the decoder attends over, trimmed to the prefix."""
        return (self.buffers["encoder_K"][:, :self.encoder_seq_len],
                self.buffers["encoder_V"][:, :self.encoder_seq_len])
