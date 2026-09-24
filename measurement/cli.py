"""Command-line conventions every measuring harness shares.

A harness passes Target construction options through as repeated
`--option key=value` (`flash_vla.inference.build`); `parse_options` turns them
into the keyword arguments `build` takes.
"""
from __future__ import annotations

from flash_vla.runtime.vla import ConfigValue


def parse_options(items: list[str]) -> dict[str, ConfigValue]:
    """`key=value` strings to a dict; true/false and integers are converted."""
    out: dict[str, ConfigValue] = {}
    for item in items:
        key, _, value = item.partition("=")
        low = value.lower()
        out[key] = (True if low == "true" else False if low == "false"
                    else int(value) if value.lstrip("-").isdigit() else value)
    return out


__all__ = ["parse_options"]
