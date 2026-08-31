"""The cold/L2 distinction, and the guard that keeps a rate honest.

protocol.md rule 6b: `valid:` must state the REGIME, not just the
configuration. This is that rule made mechanical for the units that can support
it -- a unit declaring HUT_NEEDS_COLD cannot emit a row whose walk never left
cache.

Two regimes are machine constants and one is not:
  dram  large walk, L2 flushed between timed iterations
  l2    footprint SMALLER than L2, no flush
  --    large walk WITHOUT the flush is a partly-cached DRAM read whose value
        depends on the sweep, and it inflated every absolute constant recorded
        before 2026-08-29.
"""

from __future__ import annotations

L2_BYTES = 50 * 1024 * 1024

# A walk within this multiple of L2 is not reliably cold: its result depends on
# what the PREVIOUS sweep row left resident. Two byte-identical configurations
# once read 1.51x apart for exactly this reason.
COLD_MIN_L2_RATIO = 3.0

DRAM = "dram"
L2 = "l2"


def l2_ratio(touched_bytes: int) -> float:
    return touched_bytes / L2_BYTES


def is_cold(touched_bytes: int) -> bool:
    return l2_ratio(touched_bytes) >= COLD_MIN_L2_RATIO


def guard(unit, regime: str, touched_bytes: int, *, where: str = "") -> None:
    """Refuse a measurement whose regime the unit's flags do not permit."""
    from .abi import NEEDS_COLD, NO_SOURCE
    if unit.flags & NO_SOURCE:
        return
    if regime == DRAM and unit.flags & NEEDS_COLD and not is_cold(touched_bytes):
        raise RuntimeError(
            f"{unit.name}{(' ' + where) if where else ''}: walk touches "
            f"{touched_bytes/1e6:.0f} MB = {l2_ratio(touched_bytes):.1f}x L2, "
            f"below COLD_MIN_L2_RATIO={COLD_MIN_L2_RATIO}. This is not a cold "
            f"measurement and the unit declares it needs one -- raise the bytes "
            f"moved per launch, or record it as the l2 regime instead.")


def stamp(regime: str, touched_bytes: int, flush_l2: bool) -> dict:
    """The regime fields every JSON row carries, so a constant can cite them."""
    return {"regime": regime, "touched_mb": touched_bytes / 1e6,
            "l2_ratio": l2_ratio(touched_bytes), "flush_l2": flush_l2}
