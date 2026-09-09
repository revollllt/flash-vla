"""Machine asset locations, resolved outside Target identity and captured graphs."""
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
