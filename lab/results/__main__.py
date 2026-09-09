"""Publish validated Campaign results."""
import argparse
import json
from pathlib import Path

from .publish import publish


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("publish",))
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(json.dumps(publish(args.root, args.campaign), indent=2))


if __name__ == "__main__":
    main()
