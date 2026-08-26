"""K-major XFS consumer plus the 132-CTA persistent FFN task loop.

Phase 4 host side of specs/tile/ffn_taskloop_minimal.md. Three pieces:

- `build()` compiles kernels/ffn_taskloop.cu into a plain shared library under
  the repo's .cache (shared filesystem, so a login-node build is visible to
  compute nodes) and loads it via ctypes. No torch extension machinery: the
  library has a C ABI and takes raw device pointers.
- `build_table(mode)` is the intentionally small FFN-only planner: the static
  task->CTA map with an optional dependency-gated second slot for worker reuse,
  with truncated variants for bisection --
  a persistent-kernel bug hangs rather than fails, so run-to-a-subset + compare
  is the debug loop.
- `FFNTaskloop.launch()` consumes exact-rounding K-major XFS produced directly
  by the preceding operation, resets readiness counters, then launches the
  persistent kernel.

Tensor contracts (all CUDA, contiguous): xfs_kmajor (1024, 64) bf16,
out (64, 1024) bf16, hidden (64, 4096) bf16, S (1024,) bf16,
`packed_gate_up` ((4096/32)*1024, 64) bf16 containing one
`[W_gate_tile | W_up_tile]` row per K tile, b1/b2 (4096,) bf16,
Wd (4096, 1024) bf16, g_gate (1024,) bf16, counters (32,) int32. The legacy
F and second packed-weight pointers remain in the C ABI for call-site
stability but are not read by GatedProjection. The persistent kernel consumes
XFS with four 32 KiB activation TMA operations into stationary shared memory.
`out` doubles as the residual input; the DownResidual epilogue reads each
element before writing it.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "kernels" / "ffn_taskloop.cu"
_REPO = _HERE.parents[7]  # .../backends/cuda -> src/flash_vla/... -> repo root
_CUTLASS = Path(os.environ.get("CUTLASS_DIR", "/data/user/jzou521/codes/cuda/cutlass"))
_FLASHMLA = _REPO / "third_party" / "flashmla" / "csrc"

N_CTAS_FULL = 132
FF = 4096
D = 1024
# The worker-queue experiment gives every GatedUp tile its own initial CTA
# slot. The first 32 rows carry a second, dependency-gated DownResidual task,
# so those CTAs can be reused after GatedUp completes. Rows 128..131 remain
# sentinel-idle.
GATED_UP_CTAS = 128
DOWN_RESIDUAL_TILES = 32                  # output tiles of 32 D-columns
MAX_TASKS_PER_CTA = 2
N_COUNTERS = 32
COUNTER_ARRIVE = 4

# Split-K on DownResidual's FF contraction -- must match
# DOWN_RESIDUAL_SPLIT in the kernel. Chosen
# from the measured copy floor: txns_per_warp = K_per_CTA / BK and each costs
# 248 ns [hardware-unit-test TMA-ISSUE], so S=4 takes DownResidual's 64 stages to 16
# (17.28 -> 4.32 us) and reaches the 2.78 us DRAM wall once BK is 128. S=4 also
# happens to fit the existing two-slot table exactly: 32 tiles x 4 splits = 128
# sub-tasks on the same 128 workers that already own a GatedUp tile.
DOWN_RESIDUAL_SPLIT = 4
DOWN_RESIDUAL_SUBTASKS = DOWN_RESIDUAL_TILES * DOWN_RESIDUAL_SPLIT          # 128
M_PAD = 64
BN = 32
PARTIAL_ELEMS = (DOWN_RESIDUAL_SPLIT - 1) * DOWN_RESIDUAL_TILES * M_PAD * BN   # f32 scratch

# ---------------------------------------------------------------------------
# Split-K activation-resident planner (offline-only, Phase 7)
#
# The current C ABI still consumes the compact [132, 2, 4] worker table above.
# These constants describe the next queue format without changing the default
# launch. A GatedUp tile owns one 32-column hidden tile; its 32 DownResidual partials stay on
# that worker so the hidden tile can remain in SMEM.  Thirty-two reduction
# tasks then combine the 128 partials for each output tile.
SPLITK_GATED_UP_TILES = FF // 32                 # 128 hidden tiles
SPLITK_DOWN_RESIDUAL_TILES = 1024 // 32              # 32 output tiles
SPLITK_PARTIALS_PER_GATED_UP = SPLITK_DOWN_RESIDUAL_TILES
SPLITK_MAX_TASKS_PER_CTA = 1 + SPLITK_PARTIALS_PER_GATED_UP + 1  # GatedUp + partials + reduce
SPLITK_TASK_FIELDS = 6                     # kind, column, dep, aux, id, flags
SPLITK_IDLE_CTA = 132 - SPLITK_GATED_UP_TILES   # four fixed idle workers

# Split-K task kinds.  They intentionally do not overlap the current kernel's
# kind=0 (GatedUp) and kind=1 (full DownResidual), so the compact ABI cannot accidentally
# launch this experimental schedule.
SPLITK_GATED_UP = 0
SPLITK_DOWN_RESIDUAL_PARTIAL = 2
SPLITK_REDUCE = 3


def build_splitk_plan() -> dict[str, torch.Tensor | dict[str, int]]:
    """Build the offline activation-resident split-K DAG.

    The returned tensors are CPU-side planner artifacts; the production
    ``FFNTaskloop.launch`` path deliberately rejects them because the current
    kernel has no partial-output buffer/reduction ABI yet.  Keeping the plan
    executable as a pure host artifact lets us validate the dependency graph
    and resource budgets before adding a new kernel entry point.

    ``queue`` is ``[132, 34, 6]``.  For worker ``h`` (0..127):

    - slot 0: GatedUp tile ``h`` (hidden columns ``h*32:(h+1)*32``);
    - slots 1..32: DownResidual partials for output tiles 0..31, all consuming the
      worker's local hidden tile;
    - slot 33 (workers 0..31 only): reduction for output tile ``h``.

    ``partial_deps[d]`` is the 128-arrival dependency count for reduction
    tile ``d``.  ``resources`` records the static SMEM/TMA/warp contract that
    the future kernel scheduler will use for autotuning.
    """
    queue = torch.full(
        (N_CTAS_FULL, SPLITK_MAX_TASKS_PER_CTA, SPLITK_TASK_FIELDS),
        -1, dtype=torch.int32)
    partial_deps = torch.full((SPLITK_DOWN_RESIDUAL_TILES, 2), -1, dtype=torch.int32)

    # Every GatedUp tile has its own worker. This is the key difference from the
    # rejected CTA-pair probe: no GatedUp producer is removed to make room for DownResidual.
    for h in range(SPLITK_GATED_UP_TILES):
        cta = h
        queue[cta, 0] = torch.tensor(
            [SPLITK_GATED_UP, h * 32, h // 4, h, 0, 0], dtype=torch.int32)
        for d in range(SPLITK_DOWN_RESIDUAL_TILES):
            # aux=d is the output tile, id is a globally unique partial id.
            pid = h * SPLITK_DOWN_RESIDUAL_TILES + d
            queue[cta, 1 + d] = torch.tensor(
                [SPLITK_DOWN_RESIDUAL_PARTIAL, d * 32, h, d, pid, 0], dtype=torch.int32)

    # One reducer per output tile.  It waits for 128 partials, one from every
    # hidden tile.  The reducer rows are placed on workers 0..31 after their
    # local partial list; workers 128..131 stay sentinel-idle.
    for d in range(SPLITK_DOWN_RESIDUAL_TILES):
        queue[d, 1 + SPLITK_PARTIALS_PER_GATED_UP] = torch.tensor(
            [SPLITK_REDUCE, d * 32, d, SPLITK_GATED_UP_TILES, d, 0],
            dtype=torch.int32)
        partial_deps[d] = torch.tensor([d, SPLITK_GATED_UP_TILES], dtype=torch.int32)

    resources = {
        # A 64x32 BF16 activation frame; it is local to the GatedUp worker and
        # consumed by the following partial tasks before the frame is reused.
        "partial_smem_bytes": 64 * 32 * 2,
        # One 32x32 BF16 Wd tile per partial stage.
        "partial_weight_smem_bytes": 32 * 32 * 2,
        # Float partial output is global scratch, not SMEM.  It is required to
        # avoid BF16 atomics and preserve a deterministic reduction order.
        "partial_output_bytes": SPLITK_GATED_UP_TILES * SPLITK_DOWN_RESIDUAL_TILES * 64 * 32 * 4,
        "math_warps": 4,
        "tma_warps": 2,
        "scheduler_warps": 1,
        "worker_ctas": SPLITK_GATED_UP_TILES,
    }
    validate_splitk_plan(queue, partial_deps)
    return {"queue": queue, "partial_deps": partial_deps,
            "resources": resources}


def validate_splitk_plan(queue: torch.Tensor,
                         partial_deps: torch.Tensor) -> None:
    """Validate the split-K DAG without touching CUDA state."""
    expected_shape = (N_CTAS_FULL, SPLITK_MAX_TASKS_PER_CTA, SPLITK_TASK_FIELDS)
    if queue.dtype != torch.int32 or tuple(queue.shape) != expected_shape:
        raise ValueError(f"expected split-K queue int32{expected_shape}, "
                         f"got {queue.dtype}{tuple(queue.shape)}")
    if partial_deps.dtype != torch.int32 or tuple(partial_deps.shape) != (
            SPLITK_DOWN_RESIDUAL_TILES, 2):
        raise ValueError("partial_deps must be int32[32,2]")

    # All 128 GatedUp nodes and all 4096 unique partial nodes must occur exactly
    # once.  Reductions are exactly one per output tile.
    gated_up = queue[..., 0] == SPLITK_GATED_UP
    part = queue[..., 0] == SPLITK_DOWN_RESIDUAL_PARTIAL
    red = queue[..., 0] == SPLITK_REDUCE
    if int(gated_up.sum()) != SPLITK_GATED_UP_TILES:
        raise ValueError("split-K queue must contain 128 GatedUp nodes")
    if int(part.sum()) != SPLITK_GATED_UP_TILES * SPLITK_DOWN_RESIDUAL_TILES:
        raise ValueError("split-K queue must contain 4096 partial nodes")
    if int(red.sum()) != SPLITK_DOWN_RESIDUAL_TILES:
        raise ValueError("split-K queue must contain 32 reductions")

    partial_ids = queue[..., 4][part].tolist()
    if sorted(partial_ids) != list(range(SPLITK_GATED_UP_TILES * SPLITK_DOWN_RESIDUAL_TILES)):
        raise ValueError("partial ids are not a unique 0..4095 range")
    for d in range(SPLITK_DOWN_RESIDUAL_TILES):
        row = queue[..., :][queue[..., 0] == SPLITK_REDUCE]
        r = row[d]
        if int(r[1]) != d * 32 or int(r[2]) != d or int(r[3]) != SPLITK_GATED_UP_TILES:
            raise ValueError(f"bad reduction descriptor for output tile {d}")
        if tuple(partial_deps[d].tolist()) != (d, SPLITK_GATED_UP_TILES):
            raise ValueError(f"bad reduction dependency for output tile {d}")


def _build_dir() -> Path:
    src = _SRC.read_bytes()
    tag = hashlib.sha256(src).hexdigest()[:16]
    d = _REPO / ".cache" / "cuda_ext" / f"ffn_taskloop_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build(verbose: bool = False) -> Path:
    """Compile the .so if this source hash has not been built yet."""
    out = _build_dir() / "libffn_taskloop.so"
    if out.exists():
        return out
    cuda_home = os.environ.get("CUDA_HOME", "/data/apps/cuda/13.1")
    cmd = [
        "nvcc", "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
        "-arch=sm_90a", "--expt-relaxed-constexpr",
        f"-I{_CUTLASS}/include",
        f"-I{_FLASHMLA}",
        "-o", str(out), str(_SRC),
        f"-L{cuda_home}/lib64/stubs", "-lcuda",
    ]
    if verbose:
        print("[taskloop build]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{r.stdout}\n{r.stderr}")
    return out


def build_table(mode: str = "full") -> torch.Tensor:
    """Build the fixed 132-CTA FFN task table.

    The table is ``(132, 2, 4)`` int32. Row ``i`` is the private task list
    consumed by CTA ``i``; ``kind=-1`` is an idle/sentinel row. GatedUp tiles are
    numbered 0..127 and DownResidual tiles 0..31. Four GatedUp tiles contribute to each DownResidual
    counter, matching ``COUNTER_ARRIVE`` in the kernel.

    ``mode`` is ``full``, ``gu`` or ``dr``. The latter two are compatibility
    labels for the readable GatedUp and DownResidual task kinds.
    correctness bisection and still return the same 132-CTA launch geometry.
    """
    if mode not in {"full", "gu", "dr"}:
        raise ValueError(f"unknown mode {mode!r}")

    sentinel = [-1, 0, 0, 0]
    rows = [[sentinel.copy(), sentinel.copy()] for _ in range(N_CTAS_FULL)]

    # Every GatedUp tile gets one initial worker row. In the full graph, attach one
    # DownResidual task to the second slot of the first 32 rows; the kernel executes it
    # only after the GatedUp task has released the required counter.
    if mode in ("full", "gu"):
        gated_up_tiles = [[0, tile * 32, tile // 4, 0]
                          for tile in range(128)]
        for cta, task in enumerate(gated_up_tiles):
            rows[cta][0] = task

    if mode in ("full", "dr"):
        # 32 output tiles x DOWN_RESIDUAL_SPLIT partials. Worker c owns (tile, split) =
        # divmod(c, DOWN_RESIDUAL_SPLIT), so the four splits of one tile sit on four
        # consecutive workers -- all resident, which principle 4 of
        # megakernel-taskgraph makes a correctness gate, not a preference.
        slot = 1 if mode == "full" else 0
        for cta in range(DOWN_RESIDUAL_SUBTASKS):
            tile, split = divmod(cta, DOWN_RESIDUAL_SPLIT)
            rows[cta][slot] = [1, tile * 32, 0, split]

    table = torch.tensor(rows, dtype=torch.int32)
    validate_table(table, mode)
    return table


def validate_table(table: torch.Tensor, mode: str = "full") -> None:
    """Cheap host-side checks that catch malformed schedules before launch."""
    if table.dtype != torch.int32 or tuple(table.shape) != (
            N_CTAS_FULL, MAX_TASKS_PER_CTA, 4):
        raise ValueError(
            f"expected int32[{N_CTAS_FULL},{MAX_TASKS_PER_CTA},4], got "
            f"{table.dtype}{tuple(table.shape)}")
    active = table[..., 0] >= 0
    expected = {"full": GATED_UP_CTAS + DOWN_RESIDUAL_SUBTASKS, "gu": GATED_UP_CTAS,
                "dr": DOWN_RESIDUAL_SUBTASKS}[mode]
    if int(active.sum()) != expected:
        raise ValueError(f"expected {expected} active tasks, got {int(active.sum())}")
    for row in table.tolist():
        seen_idle = False
        for kind, column, dependency, split in row:
            if kind < 0:
                seen_idle = True
                continue
            if seen_idle:
                raise ValueError("active task appears after a sentinel")
            if kind not in (0, 1):
                raise ValueError(f"invalid task kind {kind}")
            if column < 0 or column % 32:
                raise ValueError(f"invalid tile column {column}")
            if kind == 0 and not (0 <= column // 32 < 128 and
                                   0 <= dependency < N_COUNTERS):
                raise ValueError("invalid GatedUp task")
            if kind == 1 and not (0 <= column // 32 < DOWN_RESIDUAL_TILES
                                   and 0 <= split < DOWN_RESIDUAL_SPLIT):
                raise ValueError("invalid DownResidual task")


class FFNTaskloop:
    def __init__(self, verbose: bool = False):
        self._lib = ctypes.CDLL(str(build(verbose)))
        self._lib.ffn_taskloop_launch.restype = ctypes.c_int
        self._lib.ffn_taskloop_launch.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                  ctypes.c_int] + \
            [ctypes.c_void_p] * 16
        # Split-K scratch is owned here rather than passed in, so `launch`'s
        # signature -- and every call site of it -- is unchanged. Allocated on
        # the first launch, which under the bench harness is a warmup call and
        # therefore outside any graph capture.
        self._down_residual_partial = None
        self._down_residual_counters = None
        self._lib.counter_probe_launch.restype = ctypes.c_int
        self._lib.counter_probe_launch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_void_p]
        self._lib.ffn_counters_reset_launch.restype = ctypes.c_int
        self._lib.ffn_counters_reset_launch.argtypes = [ctypes.c_void_p] * 3

    def launch(self, table, xfs_kmajor, F, S, packed_gate_up, packed_gate_up_unused,
               b1, b2, Wd, g_gate,
               hidden, out, counters, *, dbg=None,
               zero_counters: bool = True,
               use_programmatic_dependency: bool = False) -> None:
        """Launch the fixed persistent FFN schedule.

        ``use_programmatic_dependency`` requires the triggering XFS producer
        to be the direct predecessor on the current stream.  In particular,
        reset the readiness counters before that producer and pass
        ``zero_counters=False`` here; otherwise the reset kernel becomes the
        immediate predecessor and the XFS/FFN overlap is lost.

        ``dbg`` is an optional host-pinned ``(n_ctas, 4)`` int64 tensor.  On a
        watchdog trap the kernel writes ``{site, g, tid, 1}`` per stuck CTA.
        """
        if tuple(table.shape) != (N_CTAS_FULL, MAX_TASKS_PER_CTA, 4):
            raise ValueError(
                f"FFN persistent launch requires table shape "
                f"({N_CTAS_FULL}, {MAX_TASKS_PER_CTA}, 4), got {tuple(table.shape)}")
        if not table.is_contiguous():
            raise ValueError("FFN task table must be contiguous")
        expected_packed_shape = ((FF // BN) * D, 2 * BN)
        if tuple(packed_gate_up.shape) != expected_packed_shape:
            raise ValueError(
                "packed_gate_up must have shape "
                f"{expected_packed_shape}, got {tuple(packed_gate_up.shape)}")
        if (not packed_gate_up.is_contiguous() or
                not packed_gate_up_unused.is_contiguous()):
            raise ValueError("packed gate/up weights must be contiguous")
        if self._down_residual_partial is None or self._down_residual_partial.device != out.device:
            self._down_residual_partial = torch.empty(PARTIAL_ELEMS, dtype=torch.float32,
                                           device=out.device)
            self._down_residual_counters = torch.zeros(DOWN_RESIDUAL_TILES, dtype=torch.int32,
                                            device=out.device)
        stream = torch.cuda.current_stream().cuda_stream
        if zero_counters:
            rc = self._lib.ffn_counters_reset_launch(
                ctypes.c_void_p(counters.data_ptr()),
                ctypes.c_void_p(self._down_residual_counters.data_ptr()),
                ctypes.c_void_p(stream))
            if rc != 0:
                raise RuntimeError(f"ffn_counters_reset_launch rc={rc}")
        if (tuple(xfs_kmajor.shape) != (D, M_PAD) or
                xfs_kmajor.dtype != torch.bfloat16 or
                not xfs_kmajor.is_contiguous()):
            raise ValueError("xfs_kmajor must be contiguous BF16 [1024,64]")
        rc = self._lib.ffn_taskloop_launch(
            ctypes.c_void_p(table.data_ptr()), int(table.shape[0]),
            int(use_programmatic_dependency),
            *[ctypes.c_void_p(t.data_ptr()) for t in
              (xfs_kmajor, F, S, packed_gate_up, packed_gate_up_unused,
               b1, b2, Wd, g_gate, hidden, out, counters,
               self._down_residual_partial, self._down_residual_counters)],
            ctypes.c_void_p(dbg.data_ptr() if dbg is not None else 0),
            ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(f"ffn_taskloop_launch rc={rc}")

    def probe(self, counters, t0s, out_ns, pairs: int) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        rc = self._lib.counter_probe_launch(
            ctypes.c_void_p(counters.data_ptr()), ctypes.c_void_p(t0s.data_ptr()),
            ctypes.c_void_p(out_ns.data_ptr()), pairs, ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(f"counter_probe_launch rc={rc}")


WATCHDOG_SITES = {
    1: "gated_up math wait full", 2: "gated_up producer wait empty",
    3: "down_residual math wait weight", 4: "down_residual math wait activation",
    5: "down_residual producer wait weight", 6: "down_residual producer wait activation",
    7: "down_residual producer counter poll",
    8: "gated_up TMA command wait", 9: "down_residual TMA command wait",
}
