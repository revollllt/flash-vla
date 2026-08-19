"""Fused wrappers for Pi0.5 call sites.

Empty for now, and that is a statement rather than a placeholder. Pi0's two
fusions -- lazy pre-norm and FlashDecoding -- are both decoder-only, and Pi0.5's
decoder is not implemented yet: its three AdaRMSNorm-bearing call sites are
blocked on the tile-dataflow spec (PLAN.md §2.4, §4.3).

Neither fusion applies to the prefix path this target currently covers. On the
encoder's shapes the norm prologue costs more than it saves: those GEMMs are
already bound on the shared memory datapath at one CTA per SM, so there are no
spare warps to absorb the extra traffic.

When the decoder lands, note that v1 deliberately does *not* carry Pi0's
`tl_fused_rms_gate` across. Its mainloop accumulates the row sum of squares from
the same shared tile the GEMM consumes, and AdaRMSNorm needs that tile unscaled
for the norm and scaled for the GEMM. v1 falls back to the separate
`tl_rms_factor` and builds on `tl_scaled_gate`, which takes the factor as a
parameter; recovering the fusion is a v2 item with a measured baseline to beat.
"""
from __future__ import annotations

FUSED_WRAPPERS: dict = {}
