"""`Pi05Inference`: weights, buffers, and three graph segments for the forward pass.

Construction binds the plan, materializes the buffer plan, loads the
checkpoint, and runs the runtime lifecycle -- warm up every segment, freeze the
scratch pool, capture each segment once. `forward` then copies the inputs in,
replays vision, tokenizes the state on the host while that runs, copies the
prompt inputs in, and replays prefix and decoder.

Two of the splits earn their keep for different reasons.

**Vision from the rest** is a data dependency. Pi0 captures everything as one
segment because its prompt is fixed at load time and `forward` is pure GPU
work. Pi0.5's prompt carries the discretized state, so every call needs ~16 us
of host tokenization before the prefix can start -- strictly serial against a
single graph. Vision depends only on the images and takes roughly 2.4 ms,
three orders of magnitude more than the host work needs, so splitting there
hides all of it.

**Prefix from decoder** is not a dependency, it is a measurement. Both could sit
in one graph; keeping them apart costs one extra launch and makes the per-stage
split readable by replaying one segment at a time. It also gives the prefix
parity gate an entry point that does not run a decoder.

Capture constrains the design the same way it does in Pi0: nothing inside a
segment may allocate. Scratch comes from a `ScratchPool`, frozen after warmup
so a missed pre-allocation raises instead of silently allocating mid-capture.
"""
from __future__ import annotations

import torch

from flash_vla.models.pi05.spec import ENCODER_LAYERS, MAX_TOKEN_LEN, runtime_shapes
from flash_vla.runtime import Identity
from flash_vla.runtime.cuda import Program, ScratchPool, Segment, StaticArena, Step

from . import pipeline
from .backends.tilelang import wrappers
from .buffers import buffer_plan
from .ops import op_table, resolve_plan
from .prefix import PrefixInputs

TARGET = "hardware/nvidia/h100/pi05"
HARDWARE = "h100-sxm5-80gb"
MODEL = "pi05"
PRECISION = "bf16"


class Pi05Inference:
    """One captured Pi0.5 forward pass.

    `steps` and `layers` exist for bisection: shortening either keeps the
    pipeline intact while cutting depth, which is how parity is read -- on random
    weights a deep run diverges chaotically between any two implementations that
    are not bit-identical.
    """

    #: The forward pass, in order: the engine protocol's `program`.
    program: tuple[Step, ...] = (
        Step("segment", "vision"),
        Step("host", "tokenize"),
        Step("segment", "prefix"),
        Step("segment", "decoder"),
    )
    #: What each segment produces, for comparison and oracle injection. The
    #: KV cache is layer-major, so its layer axis is 0.
    stage_outputs = {
        "vision": (("vision_x", None),),
        "prefix": (("prefix_K", 0), ("prefix_V", 0)),
        "decoder": (("diffusion_noise", None), ("suffix_K", 0), ("suffix_V", 0)),
    }

    def __init__(self, checkpoint, tokenizer, num_views: int, chunk_size: int,
                 steps: int = 10, layers: int = ENCODER_LAYERS, fused: bool = True,
                 prompt_len: int = MAX_TOKEN_LEN, device: str = "cuda",
                 plan: dict[str, str] | None = None):
        """`plan` maps a call-site name to the backend that implements it.

        `None` keeps every call site on TileLang -- the reference route,
        including the torch encoder-attention chain, and what the parity gates
        compare against. Nothing is overlaid on top of a plan: a call site runs a
        hand-written CUDA kernel only where the plan names it, so `self.plan` is
        the whole description of what this engine runs and a report that records
        it is reproducible. The shipped plans are named in `benchmarks/plans.py`.
        Resolved before capture, so the replay path never dispatches -- the rule
        that makes a mixed backend free at runtime.
        """
        self.num_views = num_views
        self.chunk_size = chunk_size
        self.steps = steps
        self.layers = layers
        self.prompt_len = prompt_len
        self.device = torch.device(device)
        self.plan = dict(plan) if plan else {}
        self.ops = op_table(fused, plan=self.plan or None)
        self.identity = Identity(
            target=TARGET, hardware=HARDWARE, model=MODEL,
            shape={"num_views": num_views, "chunk": chunk_size, "steps": steps,
                   "layers": layers, "prompt_len": prompt_len},
            plan=resolve_plan(self.plan or None), precision=PRECISION)

        self.weights = {
            name: torch.empty(shape, dtype=torch.bfloat16, device=device)
            for name, shape in runtime_shapes(steps).items()
        }
        plan_spec, self.encoder_seq_len = buffer_plan(num_views, chunk_size, prompt_len)
        self.arena = StaticArena(plan_spec, device)
        self.buffers = self.arena.buffers

        missing = sorted(set(self.weights) - set(checkpoint))
        if missing:
            raise KeyError(f"checkpoint is missing {missing}; run models.pi05.weights.fold")
        for name, value in checkpoint.items():
            if name in self.weights:
                self.weights[name].copy_(value)

        self.inputs = PrefixInputs(tokenizer, num_views, chunk_size)
        self.pool = ScratchPool()
        self.graphs = Program([
            Segment("vision", self._run_vision),
            Segment("prefix", self._run_prefix),
            Segment("decoder", self._run_decoder),
        ], self.pool)

    # -- task ---------------------------------------------------------------

    def set_task(self, prompt: str) -> None:
        """Install the task string. Call whenever it changes; see `Pi05Tokenizer.set_task`."""
        self.inputs.tokenizer.set_task(prompt)

    # -- segments -----------------------------------------------------------

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

    def replay(self, segment: str) -> None:
        """Replay one captured segment on the current stream."""
        self.graphs.replay(segment)

    def allocation(self, name: str) -> torch.Tensor:
        """The base allocation behind buffer `name`, padding included."""
        return self.arena.allocation(name)

    def stage(self, images: torch.Tensor, noise: torch.Tensor, **_) -> None:
        """Copy the device inputs in; the state goes through the host slot."""
        self.buffers["observation_images_normalized"].copy_(images)
        self.buffers["diffusion_noise"].copy_(noise)

    def host(self, slot: str, state=None, **_) -> None:
        """Run the one host slot: tokenize `state` and stage the prompt inputs.

        Sits between the vision and prefix replays deliberately: the vision
        graph is already executing when it runs.
        """
        if slot != "tokenize":
            raise KeyError(f"no host slot {slot!r}; this engine has 'tokenize'")
        self.inputs.build(state)
        self.inputs.copy_into(self.buffers)

    # -- inference ----------------------------------------------------------

    def sample_inputs(self, seed: int = 0) -> dict[str, torch.Tensor]:
        """Seeded random inputs at this engine's shapes: images, state, noise."""
        generator = torch.Generator(device=self.device).manual_seed(seed)

        def randn(*shape, dtype):
            return torch.randn(shape, generator=generator, device=self.device, dtype=dtype)

        return {"images": randn(self.num_views, 224, 224, 3, dtype=torch.bfloat16),
                "state": randn(32, dtype=torch.float32),
                "noise": randn(self.chunk_size, 32, dtype=torch.bfloat16)}

    def forward_prefix(self, images: torch.Tensor, state) -> int:
        """Run vision and the prefix; return the number of valid prefix rows.

        The KV cache is left in `buffers["encoder_K"]` / `["encoder_V"]`, rows
        `[:encoder_seq_len]`, of which `[:n_valid]` carry data and the rest are
        masked padding.
        """
        self.buffers["observation_images_normalized"].copy_(images)
        self.replay("vision")
        self.host("tokenize", state=state)
        self.replay("prefix")
        return self.inputs.n_valid

    def forward(self, images: torch.Tensor, state, noise: torch.Tensor) -> torch.Tensor:
        """Copy the inputs in, run every step, return the denoised chunk.

        `noise` is copied up front rather than between the prefix and decoder
        replays, so nothing on the host separates them.
        """
        self.stage(images=images, noise=noise)
        self.replay("vision")
        self.host("tokenize", state=state)
        self.replay("prefix")
        self.replay("decoder")
        return self.buffers["diffusion_noise"]

    @property
    def kv_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The prefix K and V the decoder attends over, trimmed to the prefix."""
        return self.buffers["prefix_K"], self.buffers["prefix_V"]
