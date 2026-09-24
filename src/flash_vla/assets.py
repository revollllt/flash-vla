"""Local paths of a Target's logical assets, from the machine's asset configuration.

A Target names its frozen assets (a checkpoint, a recorded fixture, an
upstream checkout) by logical IDs (`Target.assets`). Each machine maps those
IDs to local paths in one JSON file, `FLASH_VLA_ASSETS` unless a caller names
another, so the same run resolves the same assets on any machine layout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping


def resolve_assets(identifiers: Mapping[str, str], config: str | Path | None = None) -> dict[str, Path]:
    """Map role -> logical ID to local paths; relative paths use the config directory."""
    path = Path(config or os.environ["FLASH_VLA_ASSETS"]).expanduser().resolve()
    locations = json.loads(path.read_text())
    return {role: (path.parent / Path(locations[identifier]).expanduser()).resolve()
            for role, identifier in identifiers.items()}


def locate_assets(identifiers: Mapping[str, str], overrides: Mapping[str, str | None],
                  config: str | Path | None = None) -> dict[str, Path]:
    """Every role's local path: its override where one is given, its configured
    location otherwise. The configuration is read only when some role needs it."""
    explicit = {role: Path(path).expanduser().resolve()
                for role, path in overrides.items() if path is not None}
    configured = {role: identifier for role, identifier in identifiers.items()
                  if role not in explicit}
    return {**(resolve_assets(configured, config) if configured else {}), **explicit}


__all__ = ["locate_assets", "resolve_assets"]
