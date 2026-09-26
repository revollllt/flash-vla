"""Command-line conventions every measuring harness shares.

A harness names the workload with `--workload` and passes further Target
construction options as repeated `--option key=value`
(`flash_vla.inference.build`); `parse_options` turns those into the keyword
arguments `build` takes.
"""
from __future__ import annotations

from flash_vla.runtime.vla import ConfigValue


#: The help text of every harness's `--workload`.
WORKLOAD_HELP = "one of the Target's workloads (docs/workloads.md); default: the Target's first"


def parse_options(items: list[str]) -> dict[str, ConfigValue]:
    """`key=value` strings to a dict; true/false and integers are converted."""
    out: dict[str, ConfigValue] = {}
    for item in items:
        key, _, value = item.partition("=")
        low = value.lower()
        out[key] = (True if low == "true" else False if low == "false"
                    else int(value) if value.lstrip("-").isdigit() else value)
    return out


__all__ = ["WORKLOAD_HELP", "parse_options"]
