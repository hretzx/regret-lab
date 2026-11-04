from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GREY, HAIR = "#111111", "#767676", "#E5E5E5"
LINES = [(INK, "-"), ("#444444", "--"), (GREY, ":"), (INK, "-."),
         ("#999999", "-"), ("#444444", ":")]
RESULTS = Path(__file__).resolve().parent.parent / "results"


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Inter", "Arial", "DejaVu Sans"],
        "figure.facecolor": "white", "axes.facecolor": "white", "figure.figsize": (7, 3.4),
        "axes.edgecolor": HAIR, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": HAIR, "grid.linewidth": 0.6,
        "axes.labelcolor": GREY, "axes.labelsize": 7, "axes.labelpad": 8,
        "xtick.color": GREY, "ytick.color": GREY, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 1.3, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def _label_ends(ax, x_end, ends):
    y0, y1 = ax.get_ylim()
    min_gap, placed = 0.06 * (y1 - y0), None
    for y, label, color in sorted(ends):
        y_lab = y if placed is None else max(y, placed + min_gap)
        ax.text(x_end, y_lab, "  " + label, color=color, fontsize=6.5, va="center", clip_on=False)
        placed = y_lab


def plot_curves(curves, x, xlabel, ylabel, vline=None):
    apply_style()
    fig, ax = plt.subplots()
    ends = []
    for i, (label, (mean, lo, hi)) in enumerate(curves.items()):
        color, style = LINES[i % len(LINES)]
        ax.fill_between(x, lo, hi, color=GREY, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, linestyle=style, linewidth=1.3)
        ends.append((float(mean[-1]), label.upper(), color))
    ax.set_xlim(x[0], x[-1] + 0.22 * (x[-1] - x[0]))
    ax.set_ylim(bottom=0)
    if vline is not None:
        ax.axvline(vline, color="#BBBBBB", linewidth=0.8, linestyle="-")
        ax.text(vline, ax.get_ylim()[1], "  CHANGE", color=GREY, fontsize=6.5, va="top")
    _label_ends(ax, x[-1], ends)
    ax.set_xlabel(xlabel.upper())
    ax.set_ylabel(ylabel.upper())
    return fig


def save_fig(fig, name):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)
    return path
