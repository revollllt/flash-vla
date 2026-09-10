"""Plot a saved trace or rebuild historical result pages."""
import argparse
import json
from pathlib import Path

from . import plot, read_json
from .rebuild import rebuild


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rebuilding = commands.add_parser("rebuild", help="regenerate views under results/")
    rebuilding.add_argument("--root", type=Path, default=Path.cwd())
    plotting = commands.add_parser("plot", help="plot one saved trace")
    plotting.add_argument("trace", type=Path)
    plotting.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "rebuild":
        result = rebuild(args.root)
    else:
        metadata, points = plot.from_trace(read_json(args.trace))
        plot.render_optimization_progress(metadata=metadata, points=points, output_svg=args.out)
        result = {"plot": str(args.out)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
