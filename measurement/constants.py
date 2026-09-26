"""A device's measured constants, as the floor model and the transfer matrix read them.

Each hardware axis keeps a measured table (`hardware/nvidia/<device>/measured/
constants.yaml`) and names, in its `spec.ROOFLINE`, which tagged row fills
each role (`stream`, `burst`, `tensor`, ...). `load_constants` picks those
rows and versions the table, so a re-measured constant changes every report
that used it.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, NotRequired, TypedDict

#: Machine-readable fields a row may carry beside value/units/short.
ROW_FIELDS = ("fixed_us", "curve_mb_gbs", "derate_at_32")


class ConstantRow(TypedDict):
    """One tagged row of a measured table."""
    tag: str
    value: float
    units: str | None
    short: str | None
    fixed_us: NotRequired[float]
    curve_mb_gbs: NotRequired[list[list[float]]]
    derate_at_32: NotRequired[float]


@dataclass(frozen=True)
class Constants:
    """The rows a roofline names, by role; the machine's noise floor; the table's version."""
    rows: Mapping[str, ConstantRow]
    noise_floor_pct: float
    version: str

    def as_dict(self) -> dict[str, object]:
        """The report form: the noise floor beside every role's row."""
        return {"noise_floor_pct": self.noise_floor_pct, **self.rows}


def load_constants(path: Path, tags: Mapping[str, str]) -> Constants:
    """The tagged rows the ceiling uses, the machine's noise floor and the table's version."""
    import yaml
    raw = path.read_bytes()
    doc = yaml.safe_load(raw)
    rows = {row["tag"]: row for row in doc.get("constants", [])}
    picked: dict[str, ConstantRow] = {}
    for role, tag in tags.items():
        if tag not in rows:
            raise KeyError(f"constant {tag!r} ({role}) not in {path}")
        row = rows[tag]
        picked[role] = ConstantRow(tag=tag, value=row["value"], units=row.get("units"),
                                   short=row.get("short"),
                                   **{field: row[field] for field in ROW_FIELDS if field in row})
    for role, field in (("stream", "fixed_us"), ("burst", "curve_mb_gbs")):
        if role in picked and field not in picked[role]:
            raise KeyError(f"row {tags[role]!r} in {path} lacks the machine-readable {field!r}")
    return Constants(rows=picked, noise_floor_pct=float(doc["machine"]["noise_floor_pct"]),
                     version=hashlib.sha1(raw).hexdigest()[:12])


__all__ = ["ConstantRow", "Constants", "load_constants"]
