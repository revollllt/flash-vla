"""Plot an iteration table, plot a saved trace or rebuild historical pages."""
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
    curve = commands.add_parser("curve", help="plot a run directory or its CSV")
    curve.add_argument("source", type=Path,
                       help="run directory, or the iterations.csv inside one")
    curve.add_argument("--out", type=Path,
                       help="default: progress.svg beside the table")
    curve.add_argument("--title")
    curve.add_argument("--subtitle")
    curve.add_argument("--note", help="one extra line under the figure")
    curve.add_argument("--roofline-ms", type=float,
                       help="floor-model ceiling; adds the share-of-floor panel")
    curve.add_argument("--reachable-ms", type=float,
                       help="ceiling re-derived at the share this stack reaches")
    args = parser.parse_args(argv)
    if args.command == "rebuild":
        result = rebuild(args.root)
    elif args.command == "curve":
        from .curve import render
        written = render(args.source, args.out, title=args.title,
                         subtitle=args.subtitle, note=args.note,
                         roofline_ms=args.roofline_ms,
                         reachable_ms=args.reachable_ms)
        result = {"plot": str(written)}
    else:
        metadata, points = plot.from_trace(read_json(args.trace))
        plot.render_optimization_progress(metadata=metadata, points=points, output_svg=args.out)
        result = {"plot": str(args.out)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
