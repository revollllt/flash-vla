"""Host side of the Pi0.5 attention-block task loop: build, plan, validate, launch.

Mirrors `taskloop.py` (the FFN prototype). Three pieces:

- `build()` compiles kernels/attn_taskloop.cu into a plain shared library under
  the repo's .cache (shared filesystem, so a login-node build is visible to
  compute nodes) and loads it via ctypes. C ABI, raw device pointers.
- `build_table(mode)` is the offline planner: the static task -> CTA map from
  the header's dealing, with truncated variants for bisection -- a
  persistent-kernel bug hangs rather than fails, so run-a-subset + compare is
  the debug loop. `validate_table` proves the invariants the kernel assumes.
- `AttnTaskloop.launch()` zeroes the counters on-stream (graph-capturable, so a
  captured graph self-resets on replay) and launches the kernel.

Every number here is parsed from `kernels/sm90_attn_task_desc.cuh` through
`attn_reference.geometry()`, so the planner cannot drift from the ABI.
Tensor contracts: specs/tile/attention_block_contract.md.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import re
import subprocess
from pathlib import Path

import torch

from . import attn_reference as ref

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "kernels" / "attn_taskloop.cu"
_HEADER = _HERE / "kernels" / "sm90_attn_task_desc.cuh"
_REPO = _HERE.parents[7]
_CUTLASS = Path(os.environ.get("CUTLASS_DIR", "/data/user/jzou521/codes/cuda/cutlass"))
_FLASHMLA = _REPO / "third_party" / "flashmla" / "csrc"

G = ref.geometry()
N_CTAS, TASK_SLOTS = G["N_CTAS"], G["TASK_SLOTS"]
M, M_PAD, D, H, DH = G["M"], G["M_PAD"], G["D"], G["H"], G["DH"]
QKV_W, PREFIX_LEN, KEYS, KEYS_PAD = G["QKV_W"], G["PREFIX_LEN"], G["KEYS"], G["KEYS_PAD"]
QKV_TILES, QKV_SPLIT, QKV_TASKS = G["QKV_TILES"], G["QKV_SPLIT"], G["QKV_TASKS"]
ATTN_SPLIT, ATTN_TASKS = G["ATTN_SPLIT"], G["ATTN_TASKS"]
OUT_TILES, OUT_SPLIT, OUT_TASKS = G["OUT_TILES"], G["OUT_SPLIT"], G["OUT_TASKS"]
QKV_BN, OUT_BN = G["QKV_BN"], G["OUT_BN"]

COMBINE_ROWS, COMBINE_TASKS = G["COMBINE_ROWS"], G["COMBINE_TASKS"]
COMBINE_GROUPS = M_PAD // COMBINE_ROWS

KIND_QKV, KIND_ATTN, KIND_OUT, KIND_COMBINE, KIND_SENTINEL = 0, 1, 2, 3, -1
# experiment flag (see the kernel): the build consumes W_qkv pre-transposed
QKV_WEIGHT_TRANSPOSED = "ATTN_QKV_WT" in os.environ.get("ATTN_NVCC_DEFINES", "")


def _counter_map(path: Path = _HEADER) -> dict[str, int]:
    """`static constexpr int kName = expr;` members of CounterMap, evaluated."""
    decl = re.compile(r"^\s*static\s+constexpr\s+int\s+(k\w+)\s*=\s*([^;]+);", re.M)
    out: dict[str, int] = {}
    for name, expr in decl.findall(path.read_text()):
        out[name] = int(eval(expr, {"__builtins__": {}}, {**G, **out}))  # noqa: S307
    return out


CM = _counter_map()
N_COUNTERS = CM["kCount"]


def counter_for_qkv_tile(tile: int) -> int:
    """The counter a qkv n-tile releases: its head's Q counter, or the KV counter."""
    return CM["kQBegin"] + tile // (DH // QKV_BN) if tile * QKV_BN < H * DH else CM["kKv"]


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------
MODES = ("qkv", "attn", "oproj", "qkv_attn", "full")


def build_table(mode: str = "full") -> torch.Tensor:
    """Static ``(N_CTAS, TASK_SLOTS, 4)`` int32 table for one layer-step.

    The dealing is the header's: slot 0 holds the qkv tasks on CTAs
    ``[0, QKV_TASKS)`` and as many attention tasks as fit on the remaining
    CTAs (idle otherwise, so their K/V prefetch starts at t=0); the rest of
    the attention tasks take slot 1 of the split-1 qkv CTAs, which only
    publish a partial and free up first; slot 1 of the next ``COMBINE_TASKS``
    CTAs holds the combine tasks; slot 2 the o_proj tasks on ``[0, OUT_TASKS)``.
    Projection splits sit on consecutive, co-resident CTAs (split 0 waits on
    them); attention splits need not, since the combine is its own task.

    Truncated modes keep the same slot for a kind whenever the kind is present
    (so a bug that depends on slot index reproduces) and mark everything else
    sentinel; `attn` and `oproj` are run with their inputs and counters
    pre-filled by the harness.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    rows = [[[KIND_SENTINEL, 0, 0, 0] for _ in range(TASK_SLOTS)] for _ in range(N_CTAS)]
    kinds = {"qkv": (0,), "attn": (1, 3), "oproj": (2,), "qkv_attn": (0, 1, 3), "full": (0, 1, 2, 3)}[mode]
    if 0 in kinds:
        for cta in range(QKV_TASKS):
            column, split = divmod(cta, QKV_SPLIT)
            rows[cta][0] = [KIND_QKV, column, counter_for_qkv_tile(column), split]
    idle0 = N_CTAS - QKV_TASKS                       # CTAs with no slot-0 work
    late_ctas = [c for c in range(QKV_TASKS) if c % QKV_SPLIT == QKV_SPLIT - 1]
    if 1 in kinds:
        for i in range(ATTN_TASKS):
            head, split = divmod(i, ATTN_SPLIT)
            task = [KIND_ATTN, head, CM["kQBegin"] + head, split]
            if i < idle0:
                rows[QKV_TASKS + i][0] = task
            else:
                rows[late_ctas[i - idle0]][1] = task
    combine_base = max(late_ctas[:max(0, ATTN_TASKS - idle0)], default=-1) + 1
    if 3 in kinds:
        for i in range(COMBINE_TASKS):
            head, group = divmod(i, COMBINE_GROUPS)
            rows[combine_base + i][1] = [KIND_COMBINE, head, CM["kAttnBegin"] + head, group]
    if 2 in kinds:
        for cta in range(OUT_TASKS):
            column, split = divmod(cta, OUT_SPLIT)
            rows[cta][2] = [KIND_OUT, column, 0, split]
    table = torch.tensor(rows, dtype=torch.int32)
    validate_table(table, mode)
    return table


def validate_table(table: torch.Tensor, mode: str = "full") -> None:
    """Prove the invariants the kernel assumes; raise on the first violation.

    1. shape and dtype; 2. every task has exactly one owner; 3. each counter's
    producer count matches the header's arrive count; 4. the splits of a tile
    are on consecutive CTAs; 5. a slot holds only the kinds dealt to it
    (slots 0 and 1 mix kinds on disjoint CTAs).
    """
    if table.dtype != torch.int32 or tuple(table.shape) != (N_CTAS, TASK_SLOTS, 4):
        raise ValueError(f"expected int32[{N_CTAS},{TASK_SLOTS},4], got {table.dtype}{tuple(table.shape)}")
    kinds_present = {"qkv": {0}, "attn": {1, 3}, "oproj": {2}, "qkv_attn": {0, 1, 3}, "full": {0, 1, 2, 3}}[mode]
    slot_allowed = {0: {KIND_QKV, KIND_ATTN}, 1: {KIND_ATTN, KIND_COMBINE}, 2: {KIND_OUT}}
    owners: dict[tuple, int] = {}
    releases: dict[int, int] = {}
    for cta, row in enumerate(table.tolist()):
        for slot, (kind, column, dep, split) in enumerate(row):
            if kind == KIND_SENTINEL:
                continue
            if kind not in kinds_present:
                raise ValueError(f"cta {cta} slot {slot}: kind {kind} not in mode {mode!r}")
            if kind not in slot_allowed[slot]:
                raise ValueError(f"cta {cta} slot {slot}: kind {kind} not allowed in this slot")
            key = (kind, column, split)
            if key in owners:
                raise ValueError(f"task {key} owned by both cta {owners[key]} and {cta}")
            owners[key] = cta
            if kind == KIND_QKV:
                if not (0 <= column < QKV_TILES and 0 <= split < QKV_SPLIT):
                    raise ValueError(f"bad qkv task {key}")
                if dep != counter_for_qkv_tile(column):
                    raise ValueError(f"qkv tile {column} releases counter {dep}")
                if split == 0:
                    releases[dep] = releases.get(dep, 0) + 1
                if owners.get((kind, column, 0), cta) != cta - split:
                    raise ValueError(f"qkv tile {column} splits are not on consecutive CTAs")
            elif kind == KIND_ATTN:
                if not (0 <= column < H and 0 <= split < ATTN_SPLIT):
                    raise ValueError(f"bad attention task {key}")
                if dep != CM["kQBegin"] + column:
                    raise ValueError(f"attention head {column} awaits counter {dep}")
                releases[CM["kAttnBegin"] + column] = releases.get(CM["kAttnBegin"] + column, 0) + 1
            elif kind == KIND_COMBINE:
                if not (0 <= column < H and 0 <= split < COMBINE_GROUPS):
                    raise ValueError(f"bad combine task {key}")
                if dep != CM["kAttnBegin"] + column:
                    raise ValueError(f"combine head {column} awaits counter {dep}")
                releases[CM["kOBegin"] + column] = releases.get(CM["kOBegin"] + column, 0) + 1
                if owners.get((kind, column, 0), cta) != cta - split:
                    raise ValueError(f"attention head {column} splits are not on consecutive CTAs")
            else:
                if not (0 <= column < OUT_TILES and 0 <= split < OUT_SPLIT):
                    raise ValueError(f"bad o_proj task {key}")
                if owners.get((kind, column, 0), cta) != cta - split:
                    raise ValueError(f"o_proj tile {column} splits are not on consecutive CTAs")
    expected = {}
    if 0 in kinds_present:
        expected.update({KIND_QKV: QKV_TASKS})
        for h in range(H):
            if releases.get(CM["kQBegin"] + h, 0) != CM["kQArrive"]:
                raise ValueError(f"head {h} Q counter has {releases.get(CM['kQBegin'] + h, 0)} producers")
        if releases.get(CM["kKv"], 0) != CM["kKvArrive"]:
            raise ValueError("KV counter producer count mismatch")
    if 1 in kinds_present:
        expected.update({KIND_ATTN: ATTN_TASKS})
        for h in range(H):
            if releases.get(CM["kAttnBegin"] + h, 0) != CM["kAttnArrive"]:
                raise ValueError(f"head {h} attention counter has {releases.get(CM['kAttnBegin'] + h, 0)} producers")
    if 3 in kinds_present:
        expected.update({KIND_COMBINE: COMBINE_TASKS})
        for h in range(H):
            if releases.get(CM["kOBegin"] + h, 0) != CM["kOArrive"]:
                raise ValueError(f"head {h} combined-O counter has {releases.get(CM['kOBegin'] + h, 0)} producers")
    if 2 in kinds_present:
        expected.update({KIND_OUT: OUT_TASKS})
    counts = {}
    for (kind, _, _) in owners:
        counts[kind] = counts.get(kind, 0) + 1
    if counts != expected:
        raise ValueError(f"task counts {counts} != expected {expected}")
    if len(owners) > N_CTAS * TASK_SLOTS:
        raise ValueError("more tasks than slots")


def prefill_values(mode: str) -> torch.Tensor:
    """Counter image satisfying the dependencies a truncated mode's missing
    producers would have provided. CPU int32; move it to the device once and
    `counters.copy_()` it before each launch -- a device-to-device copy is
    graph-capturable where scalar slice writes are not."""
    counters = torch.zeros(N_COUNTERS, dtype=torch.int32)
    if mode in ("attn", "oproj"):
        counters[CM["kQBegin"]:CM["kQBegin"] + H] = CM["kQArrive"]
        counters[CM["kKv"]] = CM["kKvArrive"]
        counters[CM["kQkvDone"]] = CM["kQkvDoneArrive"]
    if mode == "oproj":
        counters[CM["kOBegin"]:CM["kOBegin"] + H] = CM["kOArrive"]
    return counters


def prefill_counters(mode: str, counters: torch.Tensor) -> None:
    """Eager form of `prefill_values` (not for use under graph capture)."""
    counters.copy_(prefill_values(mode).to(counters.device))


# ---------------------------------------------------------------------------
# build + launch
# ---------------------------------------------------------------------------
def _extra_flags() -> list[str]:
    """ATTN_NVCC_DEFINES: space-separated extra nvcc flags (ablation switches)."""
    return os.environ.get("ATTN_NVCC_DEFINES", "").split()


def _build_dir() -> Path:
    tag = hashlib.sha256(_SRC.read_bytes() + _HEADER.read_bytes()
                        + " ".join(_extra_flags()).encode()).hexdigest()[:16]
    d = _REPO / ".cache" / "cuda_ext" / f"attn_taskloop_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build(verbose: bool = False) -> Path:
    """Compile the .so if this source hash has not been built yet."""
    out = _build_dir() / "libattn_taskloop.so"
    if out.exists():
        return out
    cuda_home = os.environ.get("CUDA_HOME", "/data/apps/cuda/13.1")
    nvcc = os.environ.get("NVCC", "nvcc")
    cmd = [
        nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
        "-arch=sm_90a", "--expt-relaxed-constexpr", *_extra_flags(),
        f"-I{_CUTLASS}/include", f"-I{_FLASHMLA}",
        "-o", str(out), str(_SRC),
        f"-L{cuda_home}/lib64/stubs", "-lcuda",
    ]
    if verbose:
        print("[attn_taskloop build]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{r.stdout}\n{r.stderr}")
    return out


class Workspace:
    """Implementation-owned scratch (contract 3.3), allocated once per device."""

    def __init__(self, device):
        bf16, f32 = torch.bfloat16, torch.float32
        self.q_buf = torch.zeros((H, M_PAD, DH), dtype=bf16, device=device)
        self.o_buf = torch.zeros((H, M_PAD, DH), dtype=bf16, device=device)
        self.qkv_partial = torch.zeros(G["QKV_PARTIAL_F32"], dtype=f32, device=device)
        self.attn_partial = torch.zeros(G["ATTN_PARTIAL_BF16"], dtype=bf16, device=device)
        self.attn_lse = torch.zeros(G["ATTN_LSE_F32"], dtype=f32, device=device)
        self.out_partial = torch.zeros(G["OUT_PARTIAL_F32"], dtype=f32, device=device)
        self.counters = torch.zeros(N_COUNTERS, dtype=torch.int32, device=device)


class AttnTaskloop:
    def __init__(self, verbose: bool = False):
        self._lib = ctypes.CDLL(str(build(verbose)))
        self._lib.attn_taskloop_launch.restype = ctypes.c_int
        self._lib.attn_taskloop_launch.argtypes = (
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 22)
        self._lib.attn_standalone_launch.restype = ctypes.c_int
        self._lib.attn_standalone_launch.argtypes = (
            [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 22)

    def launch(self, table, ws: Workspace, *, x, rms_factor, ada_scale, w_qkv, qkv_bias,
               rope, key_mask, w_o, ada_gate, k_cache, v_cache, out,
               q_buf=None, o_buf=None, dbg=None, timeline=None,
               zero_counters: bool = True) -> None:
        """One layer-step. Mutates k_cache, v_cache, out (and the scratch).

        `q_buf` / `o_buf` default to the workspace's; pass the caller's to make
        them observable (contract 3.3). `dbg`: optional host-PINNED
        (N_CTAS, 4) int64 watchdog record; `timeline`: optional device
        (N_CTAS, TASK_SLOTS, 4) int64 buffer of per-task globaltimer stamps.
        Graph-capturable: no allocation,
        the counter reset is an on-stream memset.
        """
        if tuple(table.shape) != (N_CTAS, TASK_SLOTS, 4) or not table.is_contiguous():
            raise ValueError(f"table must be contiguous int32[{N_CTAS},{TASK_SLOTS},4]")
        q_buf = ws.q_buf if q_buf is None else q_buf
        o_buf = ws.o_buf if o_buf is None else o_buf
        shapes = {
            "x": (x, (M_PAD, D)), "rms_factor": (rms_factor, (M_PAD,)),
            "ada_scale": (ada_scale, (D,)),
            "w_qkv": (w_qkv, (QKV_W, D) if QKV_WEIGHT_TRANSPOSED else (D, QKV_W)),
            "qkv_bias": (qkv_bias, (QKV_W,)), "rope": (rope, (M_PAD, DH)),
            "key_mask": (key_mask, (KEYS_PAD,)), "w_o": (w_o, (H * DH, D)),
            "ada_gate": (ada_gate, (D,)), "k_cache": (k_cache, (KEYS_PAD, DH)),
            "v_cache": (v_cache, (KEYS_PAD, DH)), "out": (out, (M_PAD, D)),
            "q_buf": (q_buf, (H, M_PAD, DH)), "o_buf": (o_buf, (H, M_PAD, DH)),
        }
        for name, (tensor, shape) in shapes.items():
            if tuple(tensor.shape) != shape or not tensor.is_contiguous() or tensor.dtype != torch.bfloat16:
                raise ValueError(f"{name}: expected contiguous bf16{shape}, got "
                                 f"{tensor.dtype}{tuple(tensor.shape)}")
        if zero_counters:
            ws.counters.zero_()
        stream = torch.cuda.current_stream().cuda_stream
        ptr = lambda t: ctypes.c_void_p(t.data_ptr())  # noqa: E731
        rc = self._lib.attn_taskloop_launch(
            ptr(table), N_CTAS, PREFIX_LEN,
            ptr(x), ptr(rms_factor), ptr(ada_scale), ptr(w_qkv), ptr(qkv_bias), ptr(rope),
            ptr(key_mask), ptr(w_o), ptr(ada_gate), ptr(k_cache), ptr(v_cache), ptr(out),
            ptr(q_buf), ptr(o_buf),
            ptr(ws.qkv_partial), ptr(ws.attn_partial), ptr(ws.attn_lse), ptr(ws.out_partial),
            ptr(ws.counters),
            ctypes.c_void_p(dbg.data_ptr() if dbg is not None else 0),
            ctypes.c_void_p(timeline.data_ptr() if timeline is not None else 0),
            ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(f"attn_taskloop_launch rc={rc}")


STANDALONE_OPS = ("qkv_split", "qkv_reduce", "attn_split", "attn_combine", "oproj_split", "oproj_reduce",
                  "attn_combine_tok")
#: op 6 combines into `out` read as (M * H, DH) token-major -- the layout the
#: TileLang o_proj consumes -- and writes only the M real rows.
OP_ATTN_COMBINE_TOK = 6
# with no qkv split the split kernel runs its own epilogue and op 1 is not launched
STANDALONE_OP_GROUPS = {"qkv": (0, 1) if QKV_SPLIT > 1 else (0,), "attention": (2, 3), "oproj": (4, 5)}
STANDALONE_DEFAULT_OPS = STANDALONE_OP_GROUPS["qkv"] + (2, 3, 4, 5)


def _check_shapes(op: int = -1, **tensors) -> None:
    """Contract 3 shapes; `None` marks an operand the op does not read."""
    shapes = {
        "x": (M_PAD, D), "rms_factor": (M_PAD,), "ada_scale": (D,),
        "w_qkv": (QKV_W, D) if QKV_WEIGHT_TRANSPOSED else (D, QKV_W),
        "qkv_bias": (QKV_W,), "rope": (M_PAD, DH), "key_mask": (KEYS_PAD,), "w_o": (H * DH, D),
        "ada_gate": (D,), "k_cache": (KEYS_PAD, DH), "v_cache": (KEYS_PAD, DH), "out": (M_PAD, D),
        "q_buf": (H, M_PAD, DH), "o_buf": (H, M_PAD, DH),
    }
    if op == OP_ATTN_COMBINE_TOK:
        shapes["out"] = (M * H, DH)
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        shape = shapes[name]
        if tuple(tensor.shape) != shape or not tensor.is_contiguous() or tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name}: expected contiguous bf16{shape}, got "
                             f"{tensor.dtype}{tuple(tensor.shape)}")


def launch_standalone(lib, op: int, ws: "Workspace", *, x, rms_factor, ada_scale, w_qkv, qkv_bias,
                      rope, key_mask, w_o, ada_gate, k_cache, v_cache, out,
                      q_buf=None, o_buf=None, timeline=None) -> None:
    """One standalone op (see STANDALONE_OPS) on the current stream; the caller
    issues them in order.  Same tensors and scratch as `launch`; no counters.
    An operand the op does not read may be `None` (its tensor map is left
    unencoded).  Graph-capturable."""
    q_buf = ws.q_buf if q_buf is None else q_buf
    o_buf = ws.o_buf if o_buf is None else o_buf
    _check_shapes(op, x=x, rms_factor=rms_factor, ada_scale=ada_scale, w_qkv=w_qkv, qkv_bias=qkv_bias,
                  rope=rope, key_mask=key_mask, w_o=w_o, ada_gate=ada_gate, k_cache=k_cache,
                  v_cache=v_cache, out=out, q_buf=q_buf, o_buf=o_buf)
    stream = torch.cuda.current_stream().cuda_stream
    ptr = lambda t: ctypes.c_void_p(t.data_ptr() if t is not None else 0)  # noqa: E731
    rc = lib.attn_standalone_launch(
        int(op), PREFIX_LEN,
        ptr(x), ptr(rms_factor), ptr(ada_scale), ptr(w_qkv), ptr(qkv_bias), ptr(rope),
        ptr(key_mask), ptr(w_o), ptr(ada_gate), ptr(k_cache), ptr(v_cache), ptr(out),
        ptr(q_buf), ptr(o_buf),
        ptr(ws.qkv_partial), ptr(ws.attn_partial), ptr(ws.attn_lse), ptr(ws.out_partial),
        ptr(ws.counters), ctypes.c_void_p(0),
        ctypes.c_void_p(timeline.data_ptr() if timeline is not None else 0),
        ctypes.c_void_p(stream))
    if rc != 0:
        raise RuntimeError(f"attn_standalone_launch op={op} rc={rc}")


WATCHDOG_SITES = {
    1: "qkv math wait full", 2: "qkv producer wait empty", 3: "qkv split-0 join",
    4: "attention math wait full", 5: "attention producer wait empty",
    6: "attention producer dependency (Q/KV counter)", 7: "attention split-0 join",
    8: "o_proj math wait full", 9: "o_proj producer wait empty",
    10: "o_proj producer dependency (head combined)", 11: "o_proj split-0 join",
    12: "o_proj split-0 wait qkv-done (x alias)",
}
