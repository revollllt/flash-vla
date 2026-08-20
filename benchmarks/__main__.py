"""CLI entry point: python -m benchmarks <command> [options]."""
from __future__ import annotations

import sys

COMMANDS = {"e2e": "e2e", "e2e-pi05": "e2e_pi05", "profile": "profile", "profile-pi05": "profile_pi05",
            "kernels": "kernels"}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print(f"commands: {', '.join(COMMANDS)}")
        print("run `python -m benchmarks <command> --help` for options")
        return 1 if argv else 0
    from importlib import import_module

    module = import_module(f".{COMMANDS[argv[0]]}", __package__)
    return module.main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
