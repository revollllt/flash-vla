"""Two-panel progress figure for a new optimization run's iteration table.

Panel A places each retained measurement at the wall-clock time it was taken,
which shows the cadence of the run; panel B puts the same points in milestone
order under their short names, which shows what was actually done. Both read one
`iterations.csv`, so a run only has to supply data, a short label per row and a
title.

Columns beyond the table's required ones, all optional:

    label    short name for panel B's axis; falls back to a trimmed `change`
    group    colour series, e.g. the segment a change touched
    callout  annotate this point in panel A; leave blank for the rest

Times come from each row's benchmark JSON (`measurement_context.timestamp`).
When none resolve, panel A is dropped rather than faked.
"""
import csv
import json
from datetime import datetime
from pathlib import Path
from textwrap import fill

#: Retained models. Only these advance the line; the rest are trials.
RETAINED = {"start", "keep"}

#: Colour-blind-safe and legible at small sizes, in assignment order.
PALETTE = ["#1f7a8c", "#8e6bab", "#d97a28", "#1a8f6a", "#b2456e", "#6b7280"]
INK, MUTED, RULE = "#1f2933", "#5b6672", "#d7dce2"


def _rows(table):
    with Path(table).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("the iteration table contains no measurements")
    return rows


def _timestamp(row, root):
    """The trial's measurement time, or None when its report is absent."""
    report = (row.get("report") or "").strip()
    if not report:
        return None
    path = root / report
    if not path.is_file():
        return None
    try:
        stamp = json.loads(path.read_text())["measurement_context"]["timestamp"]
    except (KeyError, ValueError):
        return None
    return datetime.fromtimestamp(float(stamp))


def _milestones(rows, root):
    """Retained measurements, in order, with everything the figure needs."""
    out, groups = [], {}
    for row in rows:
        if row["decision"] not in RETAINED or not row["latency_ms"]:
            continue
        group = (row.get("group") or "").strip() or "run"
        groups.setdefault(group, PALETTE[len(groups) % len(PALETTE)])
        label = (row.get("label") or "").strip() or row["change"][:34].rstrip(" ,.")
        out.append({
            "iteration": int(row["iteration"]),
            "ms": float(row["latency_ms"]),
            "label": label,
            "group": group,
            "colour": groups[group],
            "callout": (row.get("callout") or "").strip(),
            "when": _timestamp(row, root),
        })
    if not out:
        raise ValueError("no retained measurement carries a latency")
    return out, groups


def _trials(rows, root, groups):
    """Measured trials that did not advance the line, drawn but not connected."""
    return [{"ms": float(row["latency_ms"]), "when": _timestamp(row, root),
             "iteration": int(row["iteration"]),
             "colour": groups.get((row.get("group") or "").strip(), MUTED)}
            for row in rows
            if row["decision"] not in RETAINED and row["latency_ms"]]


def _solid_runs(points):
    """Maximal runs of consecutive iterations.

    The bridging point is deliberately NOT carried into the preceding run: a
    solid stroke drawn across a gap would sit on top of the dashed one and hide
    the fact that trials were measured and rejected in between.
    """
    runs, run = [], [points[0]]
    for previous, point in zip(points, points[1:]):
        if point["iteration"] == previous["iteration"] + 1:
            run.append(point)
        else:
            runs.append(run)
            run = [point]
    runs.append(run)
    return [run for run in runs if len(run) > 1]


def _draw_line(ax, points, key):
    """The retained line: solid within a run of consecutive iterations, dashed
    across a gap where rejected trials sit."""
    for previous, point in zip(points, points[1:]):
        if point["iteration"] != previous["iteration"] + 1:
            ax.plot([previous[key], point[key]], [previous["ms"], point["ms"]],
                    color="#95a0ad", linewidth=1.5, linestyle=(0, (3.5, 3)),
                    zorder=1)
    for run in _solid_runs(points):
        ax.plot([p[key] for p in run], [p["ms"] for p in run],
                color="#95a0ad", linewidth=1.8, zorder=1,
                solid_capstyle="round")


def _callouts(ax, points, key):
    """Annotate the rows that asked for it.

    The side is taken from where the point sits, because text placed outward
    from a point in the last third of the axis runs off the figure; the
    vertical offset cycles so two callouts close in x do not stack.
    """
    marked = [p for p in points if p["callout"]]
    if not marked:
        return
    span = [p[key] for p in points]
    low, high = min(span), max(span)
    reach = (high - low) or 1
    for index, point in enumerate(marked):
        fraction = (point[key] - low) / reach
        left = fraction > 0.55
        dx = -30 if left else 30
        # Always upward. A retained-latency curve descends, so the band above
        # it is the empty one; placing a callout below runs it into the axis
        # floor exactly where the curve is flattest and the points crowd.
        dy = (36, 66, 46, 86)[index % 4]
        ax.annotate(
            f'{point["ms"]:.3f}  {fill(point["callout"], 24)}',
            xy=(point[key], point["ms"]), xytext=(dx, dy),
            textcoords="offset points", fontsize=8.5, color=point["colour"],
            ha="right" if left else "left", linespacing=1.4, zorder=5,
            arrowprops=dict(arrowstyle="-", linewidth=0.9,
                            color=point["colour"], shrinkA=2, shrinkB=5,
                            connectionstyle="angle,angleA=0,angleB=75,rad=0"))


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(axis="y", color=RULE, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def render(table, output, *, title=None, subtitle=None, note=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    from matplotlib.lines import Line2D

    table, output = Path(table), Path(output)
    rows = _rows(table)
    points, groups = _milestones(rows, table.parent)
    trials = _trials(rows, table.parent, groups)
    timed = [p for p in points if p["when"] is not None]
    dual = len(timed) >= 2

    first, last = points[0]["ms"], points[-1]["ms"]
    heading = title or f'{first:.3f} → {last:.3f} ms'
    attempted = sum(1 for row in rows if row["decision"] not in {"start"})
    rejected = sum(1 for row in rows if row["decision"] in {"revert", "failed"})

    height = 11.6 if dual else 6.6
    fig = plt.figure(figsize=(12.6, height), layout="constrained")
    top = 0.875 if dual else 0.80
    fig.get_layout_engine().set(rect=(0.010, 0.042, 0.982, top - 0.042))
    grid = fig.add_gridspec(2 if dual else 1, 1,
                            height_ratios=[1.0, 1.2] if dual else [1.0])

    fig.text(0.010, 0.985, heading, ha="left", va="top", fontsize=21,
             weight="bold", color=INK)
    lines = [subtitle] if subtitle else []
    lines.append(f'{len(points)} retained of {attempted} trials, {rejected} rejected'
                 f' · one measurement condition throughout')
    fig.text(0.010, 0.945, "\n".join(lines), ha="left", va="top", fontsize=10,
             color=MUTED, linespacing=1.5)

    if len(groups) > 1:
        fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=c,
                                   markersize=6, label=g)
                            for g, c in groups.items()],
                   loc="upper left",
                   bbox_to_anchor=(0.010, top + 0.028 if dual else 0.86),
                   ncols=len(groups), frameon=False, fontsize=9.5,
                   labelcolor=MUTED, handletextpad=0.4, columnspacing=1.8)

    # -- Panel A: when the run actually happened ----------------------------
    if dual:
        ax = fig.add_subplot(grid[0])
        _style(ax)
        _draw_line(ax, timed, "when")
        for point in timed:
            ax.scatter(point["when"], point["ms"], s=42, color=point["colour"],
                       zorder=3, edgecolor="white", linewidth=0.8)
        for trial in trials:
            if trial["when"] is not None:
                ax.scatter(trial["when"], trial["ms"], s=34, facecolor="none",
                           edgecolor=trial["colour"], linewidth=1.2, zorder=2)
        _callouts(ax, timed, "when")
        ax.set_title("A   Measurement time", loc="left", fontsize=11.5,
                     weight="bold", color=INK, pad=10)
        ax.set_ylabel("deployed end-to-end median (ms)", fontsize=9.5,
                      color=MUTED)
        ax.xaxis.set_major_formatter(DateFormatter("%b %d\n%H:%M"))
        ax.margins(x=0.10, y=0.30)

    # -- Panel B: what was done, in order -----------------------------------
    ax = fig.add_subplot(grid[1] if dual else grid[0])
    _style(ax)
    order = list(range(len(points)))
    for point, x in zip(points, order):
        point["order"] = x
    _draw_line(ax, points, "order")
    for point in points:
        ax.scatter(point["order"], point["ms"], s=42, color=point["colour"],
                   zorder=3, edgecolor="white", linewidth=0.8)
        ax.annotate(f'{point["ms"]:.3f}', xy=(point["order"], point["ms"]),
                    xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8.5, color=point["colour"], weight="bold")
    #: A trial sits between the retained models it was measured against.
    by_iteration = {p["iteration"]: p["order"] for p in points}
    for trial in trials:
        lower = [o for i, o in by_iteration.items() if i < trial["iteration"]]
        if lower:
            ax.scatter(max(lower) + 0.5, trial["ms"], s=34, facecolor="none",
                       edgecolor=trial["colour"], linewidth=1.2, zorder=2)
    ax.set_xticks(order)
    ax.set_xticklabels([f'{p["iteration"]:02d}  {p["label"]}' for p in points],
                       rotation=32, ha="right", fontsize=8.5, color=MUTED)
    ax.set_title("B   Milestone order", loc="left", fontsize=11.5,
                 weight="bold", color=INK, pad=10)
    ax.set_ylabel("deployed end-to-end median (ms)", fontsize=9.5, color=MUTED)
    ax.margins(x=0.04, y=0.20)

    footer = ["Solid: consecutive retained models. Dashed: a gap where trials "
              "were measured and rejected. Hollow: measured, not retained."]
    if note:
        footer.append(note)
    fig.text(0.010, 0.006, "\n".join(footer), ha="left", va="bottom",
             fontsize=8.5, color=MUTED, linespacing=1.6)

    output.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.hashsalt": "flash-vla"}):
        fig.savefig(output, format=output.suffix.lstrip(".") or "svg")
    plt.close(fig)
    return output
