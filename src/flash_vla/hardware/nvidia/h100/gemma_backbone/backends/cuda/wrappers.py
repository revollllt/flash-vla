"""Hand-written CUDA call sites for the Gemma backbone, shared by both H100 Targets.

The backbone is 18 layers of a 2048-wide transformer over the prefix, and both
Pi0.5 and Pi0 run it at the same widths; only the row count differs (968 with
Pi0.5's 200-token prompt, 768 with Pi0's empty one). Every wrapper here takes
the row count from the tensors it is handed, so one build serves both.

Call sites a plan can route here:

    llm_backbone_attention          the fused MQA prefix attention kernel
    llm_backbone_norm_gated_ffn     an RMSNorm launch and one persistent
                                    warp-specialized dual-GEMM
    llm_backbone_out_proj_residual  cuBLAS `addmm_`
    llm_backbone_ffn_down_residual  cuBLAS `addmm_`

The two residual sites are cuBLAS on purpose, not for want of a kernel. Their
epilogue -- `out += x @ W` with fp32 accumulation and a bf16 in-place residual
-- is exactly cuBLAS's contract, a hand-written sm90 GEMM lost to it at these
shapes once already
(`.agents/notes/rejected/architecture/2026-09-03-sm90-short-k-gemm.md`), and
Pi0.5 has routed them to `addmm_` since
`.agents/notes/implemented/architecture/2026-09-03-prefix-gemm-epilogue.md`.
What this backend adds is that Pi0 can reach them too: they live in a
component package rather than in one Target's wrappers.

No call site here declares a route constraint. Every wrapper reads and writes
the graph's own buffers in the layout the TileLang route uses -- in particular
Q stays token-major `(rows * heads, head_dim)` rather than the head-major
scratch the action expert's CUDA pair keeps -- so a plan may route any one of
these alone, and `llm_backbone_norm_qkv_rope` may stay on another backend.
"""
from __future__ import annotations

import os

import torch

from . import enc_attn as _enc
from . import gated_ffn as _gu


class _AttnState:
    """The attention library and the key mask, built when a plan routes here."""

    def __init__(self, scratch):
        self.scratch = scratch
        self.library_built = False
        self.verbose = bool(os.environ.get("FLASH_VLA_BUILD_VERBOSE"))

    def key_mask(self, keys: int, device) -> torch.Tensor:
        """An all-zero additive mask, for a Target whose prefix needs none.

        Pi0's prefix is 768 rows, twelve whole 64-key blocks with no padding
        and no masked positions, so its graph passes `mask=None`. The kernel
        still wants a pointer: it pads the mask into shared memory itself and
        substitutes a large negative beyond `keys`, so all this has to supply
        is `keys` zeros. It comes from the runner's allocator, which zeroes on
        allocation and freezes after warmup, so the buffer is allocated on the
        first (warmup) call and reused unchanged by every graph replay.
        """
        return self.scratch("gemma_attn_key_mask", (keys,), torch.bfloat16, device)


def llm_backbone_attention(state: _AttnState, Q, K, V, scale, mask, out):
    """out = softmax(Q @ K.T * scale + mask) @ V over the whole prefix, one launch.

    `Q` is (rows * heads, head_dim) token-major, `K`/`V` are (keys, head_dim)
    leading-row views of the layer's KV cache, `mask` is (keys,) additive bf16
    or `None`, `out` is (rows * heads, head_dim) and is fully written. Safe
    during CUDA-graph capture: the tensor maps are encoded on the first call,
    which warmup makes, and cached on the buffer pointers.
    """
    if mask is None:
        mask = state.key_mask(K.shape[0], K.device)
    _enc.attention(Q, K, V, scale, mask, out, verbose=state.verbose)
    return out


def llm_backbone_norm_gated_ffn(state: "_GatedFfnState", x, gate_w, up_w, out, x_norm):
    """out = gelu_tanh(x_norm @ gate_w) * (x_norm @ up_w), with x_norm materialized.

    Two launches: the RMSNorm, then the persistent dual-GEMM. `x` and `x_norm`
    are (rows, 2048), the weights (2048, 16384), `out` (rows, 16384) written in
    full. `x_norm` is the graph's own buffer and the op spec declares it an
    auxiliary output, so it is written rather than kept in registers -- and it
    has to be, because the projection must see the bf16-rounded activation.
    """
    rows = x.shape[0]
    _gu.rms_norm(x, x_norm[:rows], verbose=state.verbose)
    _gu.gated_ffn(x_norm[:rows], gate_w, up_w, out[:rows],
                  sm_count=state.sm_count, verbose=state.verbose)
    return out


def llm_backbone_out_proj_residual(x, weight, out):
    """out += attn @ weight, in place: cuBLAS, fp32 accumulation, bf16 residual.

    `x` is (rows, 2048) -- the attention result the graph reads as token-major
    -- `weight` is (2048, 2048), and `out` is both the residual and the
    result. No allocation, graph-capturable.
    """
    out.addmm_(x, weight)
    return out


def llm_backbone_ffn_down_residual(x, weight, out):
    """out += hidden @ weight, in place; same contract, K = 16384.

    Provided so a plan may select it, but **Pi0's candidate plan deliberately
    does not**, and Pi0.5 already reaches this code through its own TileLang
    wrapper. Measured in the graph on Pi0, both plans in one job (599832),
    cuBLAS is 6.61 us per call SLOWER than Pi0's TileLang `tl_matmul_res_ws`
    body: 91.98 against 85.37 us, +112 us over 17 layers. Two isolated timers
    had said the opposite, and the regime is why. In the pipeline the 25 MB
    hidden buffer this reads was written by the gated FFN one kernel earlier
    and is L2-resident; the TileLang body carries `SWIZZLE=8` to rasterize for
    exactly that reuse, and cuBLAS chooses its own order. Pi0 also leaves 36 of
    132 SMs idle here (6 x 16 = 96 tiles at a 128x128 tile), which caps both
    implementations at about two thirds of the wgmma ceiling.
    """
    out.addmm_(x, weight)
    return out


class _GatedFfnState:
    """The gated-FFN library handle and the grid it launches, built once per table.

    The SM count is read on the first call, not at construction: `eval.smoke`
    builds every Target's op table on a login node with no GPU, so a backend
    that touches the device while a plan is merely being declared cannot be
    checked there.
    """

    def __init__(self) -> None:
        self.verbose = bool(os.environ.get("FLASH_VLA_BUILD_VERBOSE"))
        self._sm_count: int | None = None

    @property
    def sm_count(self) -> int:
        """The persistent kernel's grid: one CTA per SM, which its 224 KB pool forces anyway."""
        if self._sm_count is None:
            self._sm_count = torch.cuda.get_device_properties(
                torch.cuda.current_device()).multi_processor_count
        return self._sm_count


#: The call sites this backend implements.
NAMES = frozenset({
    "llm_backbone_attention",
    "llm_backbone_norm_gated_ffn",
    "llm_backbone_out_proj_residual",
    "llm_backbone_ffn_down_residual",
})

#: No extension ops: every call site is in the standard vocabulary.
OPS: tuple = ()


def make_wrappers(scratch, selected_names=None) -> dict:
    """Build the wrappers a plan routed here, sharing one library and workspace."""
    names = set(selected_names) if selected_names is not None else set(NAMES)
    unknown = names - set(NAMES)
    if unknown:
        raise KeyError(f"gemma_backbone.cuda does not implement {sorted(unknown)}; "
                       f"it provides {sorted(NAMES)}")
    table: dict = {}
    if "llm_backbone_attention" in names:
        state = _AttnState(scratch)
        table["llm_backbone_attention"] = (
            lambda Q, K, V, scale, mask, out:
            llm_backbone_attention(state, Q, K, V, scale, mask, out))
    if "llm_backbone_norm_gated_ffn" in names:
        gu_state = _GatedFfnState()
        table["llm_backbone_norm_gated_ffn"] = (
            lambda x, gate_w, up_w, out, x_norm:
            llm_backbone_norm_gated_ffn(gu_state, x, gate_w, up_w, out, x_norm))
    if "llm_backbone_out_proj_residual" in names:
        table["llm_backbone_out_proj_residual"] = llm_backbone_out_proj_residual
    if "llm_backbone_ffn_down_residual" in names:
        table["llm_backbone_ffn_down_residual"] = llm_backbone_ffn_down_residual
    return table


__all__ = ["NAMES", "OPS", "make_wrappers"]
