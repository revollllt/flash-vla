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
    """Annotate the rows that asked for it, in lanes across the top.

    Offsets measured from each point do not work here: two points at different
    heights can be pushed to the same absolute height and collide. The text is
    therefore placed in axes fractions -- a small set of lanes below the top
    edge -- with a leader down to the point, which also keeps it inside the
    axes. The side is taken from where the point sits, so text near the right
    edge reads inward.
    """
    from datetime import datetime as _datetime
    from matplotlib.dates import date2num

    marked = [p for p in points if p["callout"]]
    if not marked:
        return
    left_edge, right_edge = ax.get_xlim()
    reach = (right_edge - left_edge) or 1
    lanes = (0.98, 0.80, 0.89, 0.71)
    for index, point in enumerate(marked):
        value = point[key]
        numeric = date2num(value) if isinstance(value, _datetime) else value
        fraction = (numeric - left_edge) / reach
        left = fraction > 0.55
        text_x = min(max(fraction + (-0.03 if left else 0.03), 0.015), 0.985)
        ax.annotate(
            f'{point["ms"]:.3f}  {fill(point["callout"], 24)}',
            xy=(value, point["ms"]), xycoords="data",
            xytext=(text_x, lanes[index % len(lanes)]),
            textcoords="axes fraction", fontsize=8.5, color=point["colour"],
            ha="right" if left else "left", va="top", linespacing=1.4, zorder=5,
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


def _order_axis(ax, points, order):
    """Shared x axis of the milestone panels: one tick per retained model."""
    ax.set_xticks(order)
    ax.set_xticklabels([f'{p["iteration"]:02d}  {p["label"]}' for p in points],
                       rotation=32, ha="right", fontsize=8.5, color=MUTED)


def render(table, output, *, title=None, subtitle=None, note=None,
           roofline_ms=None, reachable_ms=None):
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

    share = roofline_ms is not None
    height = (13.4 if share else 11.6) if dual else (8.4 if share else 6.6)
    fig = plt.figure(figsize=(12.6, height), layout="constrained")
    top = (0.895 if share else 0.875) if dual else 0.84
    fig.get_layout_engine().set(rect=(0.010, 0.042, 0.982, top - 0.042))
    rows_of = (1 if dual else 0) + 1 + (1 if share else 0)
    ratios = ([1.0] if dual else []) + [1.15] + ([0.72] if share else [])
    grid = fig.add_gridspec(rows_of, 1, height_ratios=ratios)

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
        ax.set_title("A   Measurement time", loc="left", fontsize=11.5,
                     weight="bold", color=INK, pad=10)
        ax.set_ylabel("deployed end-to-end median (ms)", fontsize=9.5,
                      color=MUTED)
        ax.xaxis.set_major_formatter(DateFormatter("%b %d\n%H:%M"))
        ax.margins(x=0.10, y=0.34)
        ax.autoscale_view()
        _callouts(ax, timed, "when")

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
    if not share:
        _order_axis(ax, points, order)
    ax.set_title("B   Milestone order", loc="left", fontsize=11.5,
                 weight="bold", color=INK, pad=10)
    ax.set_ylabel("deployed end-to-end median (ms)", fontsize=9.5, color=MUTED)
    ax.margins(x=0.04, y=0.20)
    order_ax = ax

    # -- Panel C: the same points against the floor ------------------------
    if share:
        ax = fig.add_subplot(grid[rows_of - 1], sharex=order_ax)
        _style(ax)
        pct = [roofline_ms / p["ms"] * 100 for p in points]
        ax.plot(order, pct, color="#95a0ad", linewidth=1.8, zorder=1,
                solid_capstyle="round")
        for point, x, value in zip(points, order, pct):
            ax.scatter(x, value, s=42, color=point["colour"], zorder=3,
                       edgecolor="white", linewidth=0.8)
        for index in (0, len(points) - 1):
            ax.annotate(f'{pct[index]:.0f}%', xy=(order[index], pct[index]),
                        xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=9, weight="bold", color=points[index]["colour"])
        if reachable_ms:
            # The model divides compute-bound sites by 100% of the tensor peak,
            # which nothing in this route reaches; this is the same ceiling
            # re-derived at the share the stack actually delivers.
            limit = roofline_ms / reachable_ms * 100
            ax.axhline(limit, color="#b2456e", linewidth=1.1,
                       linestyle=(0, (5, 3)), zorder=2)
            ax.annotate(f'{limit:.0f}%  reachable floor, {reachable_ms:.2f} ms',
                        xy=(order[-1], limit), xytext=(-4, 7),
                        textcoords="offset points", ha="right", fontsize=8.5,
                        color="#b2456e")
        _order_axis(ax, points, order)
        order_ax.tick_params(labelbottom=False)
        ax.set_title("C   Share of the floor reached", loc="left",
                     fontsize=11.5, weight="bold", color=INK, pad=10)
        ax.set_ylabel(f"% of the {roofline_ms:.3f} ms ceiling", fontsize=9.5,
                      color=MUTED)
        ax.margins(x=0.04, y=0.28)

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
