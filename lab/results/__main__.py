"""Publish, validate or rebuild Campaign results."""
import argparse
import json
from pathlib import Path

from .publish import publish
from .rebuild import rebuild, validate_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publishing = commands.add_parser("publish")
    publishing.add_argument("campaign", type=Path)
    rebuilding = commands.add_parser("rebuild")
    rebuilding.add_argument("--check", action="store_true")
    validating = commands.add_parser("validate")
    for command in (publishing, rebuilding, validating):
        command.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.command == "publish":
        result = publish(args.root, args.campaign)
    elif args.command == "rebuild":
        result = rebuild(args.root, check=args.check)
    else:
        result = validate_results(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
