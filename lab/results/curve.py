"""Plot a small iteration table for a new optimization run."""
import csv
from pathlib import Path
from textwrap import fill


def render(table, output, *, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    table, output = Path(table), Path(output)
    with table.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("the iteration table contains no measurements")

    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    positions, retained = [], []
    current = float("nan")
    labels = set()
    for row in rows:
        iteration = int(row["iteration"])
        latency = float(row["latency_ms"]) if row["latency_ms"] else None
        decision = row["decision"]
        if decision in {"start", "keep"}:
            current = float(row["latency_ms"])
        positions.append(iteration)
        retained.append(current)
        if latency is not None:
            kept = decision in {"start", "keep"}
            color = "#167d8d" if kept else "#8a647c"
            label = "Retained measurement" if kept else "Other trial"
            ax.scatter(iteration, latency, s=45, marker="o" if kept else "x",
                       color=color, zorder=3, label=label if label not in labels else None)
            labels.add(label)
            if kept or len(rows) <= 12:
                suffix = "" if kept else f" · {decision}"
                text = f'{latency:.3f} ms{suffix}\n{fill(row["change"], 24)}'
                ax.annotate(text, (iteration, latency), xytext=(0, 12),
                            textcoords="offset points", ha="center", fontsize=8, color="#334155",
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
        else:
            ax.annotate(f'{decision}\n{fill(row["change"], 22)}', (iteration, 0.025),
                        xycoords=ax.get_xaxis_transform(), ha="center", fontsize=8, color="#8a647c")

    ax.plot(positions, retained, color="#167d8d", linewidth=2, alpha=0.8,
            drawstyle="steps-post", label="Retained model")
    ax.set_title(title or table.parent.name, loc="left", fontsize=16, fontweight="bold", pad=18)
    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel("Deployed end-to-end latency · median (ms)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.margins(x=0.08, y=0.3)
    ax.legend(frameon=False, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Date": None})
    plt.close(fig)
