"""Figures of the final report, drawn from ``results.py`` and the saved run outputs, written as SVG."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import results as R  # noqa: E402

HERE = Path(__file__).parent
REPO = HERE.parent
INTERPRETABILITY = REPO / "outputs/soft_quantile_monokan/interpretability"

BLUE, DARK, LIGHT, GREY, ORANGE, GREEN = "#0065BD", "#003359", "#98C6EA", "#7F7F7F", "#E37222", "#A2AD00"
COLORS = {"monokan": BLUE, "kan": ORANGE, "fcnn": GREY, "perfect": "#333333", "mlp": GREY}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "svg.fonttype": "none",
})


def _save(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def violations(out_dir: Path) -> None:
    """Main design: how often the stock rule breaks, and what a break looks like."""
    curves = json.loads((HERE / "data/violation_curves.json").read_text(encoding="utf-8"))
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1.35]})

    names = ["KAN, seed 2025", "KAN, seed 2026", "KAN, seed 2027", "MonoKAN, all seeds", "FCNN, all seeds"]
    values = [*R.KAN_VIOLATION_BY_SEED.values(), 0.0, 0.0]
    bars = left.barh(names[::-1], values[::-1], color=[GREY, BLUE, ORANGE, ORANGE, ORANGE], height=0.6)
    for bar, value in zip(bars, values[::-1]):
        left.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center",
                  fontsize=8.5)
    left.set_xlim(0, 62)
    left.set_xlabel("products that order more when stock rises")
    left.set_title("(a) Stock-rule breaks per training run")
    left.grid(axis="y", visible=False)

    stock = np.array(curves["stock"])
    right.plot(stock, curves["perfect"], color=COLORS["perfect"], lw=1.4, ls="--", label="Optimal policy")
    right.plot(stock, curves["kan"], color=ORANGE, lw=1.8, label="KAN without the certificate")
    right.plot(stock, curves["monokan"], color=BLUE, lw=1.8, label="MonoKAN")
    kan = np.array(curves["kan"])
    low = int(np.argmin(np.where((stock > 0.3) & (stock < 1.0), kan, np.inf)))
    high = int(np.argmax(np.where((stock > low * 0.025) & (stock < 1.6), kan, -np.inf)))
    right.axvspan(stock[low], stock[high], color=ORANGE, alpha=0.08, lw=0)
    right.annotate("KAN orders more\nas stock rises", xy=(stock[(low + high) // 2], kan[(low + high) // 2]),
                   xytext=(1.55, 0.95), fontsize=8, color=ORANGE,
                   arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 0.8})
    right.set_xlim(0, 4)
    right.set_ylim(bottom=0)
    right.set_xlabel("stock on hand (days of mean demand)")
    right.set_ylabel("order (days of mean demand)")
    right.set_title("(b) One product: order as stock rises")
    right.legend(loc="upper right", fontsize=7.8)
    fig.tight_layout(w_pad=2.5)
    _save(fig, out_dir, "fig_violations")


def reaction_strength(out_dir: Path) -> None:
    """Main design: the direction is guaranteed, the strength is not."""
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    labels = ["Optimal", "Same KAN,\nno certificate", "MonoKAN", "Paper FCNN"]
    values = [abs(R.OMNISCIENT_SLOPE), abs(R.DIRECT["kan"]["slope"]), abs(R.DIRECT["monokan"]["slope"]),
              abs(R.DIRECT["fcnn"]["slope"])]
    bars = ax.bar(labels, values, color=[COLORS["perfect"], ORANGE, BLUE, GREY], width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=8.5)
    ax.axhline(1.0, color=GREY, lw=0.8, ls=":")
    ax.text(3.45, 1.01, "one-for-one", fontsize=7.5, color=GREY, ha="right", va="bottom")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("order reduction per unit of stock")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=8)
    _save(fig, out_dir, "fig_reaction")


def development(out_dir: Path) -> None:
    """Alternative design: regret after each cumulative change."""
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    labels = [label for label, _ in R.DEVELOPMENT]
    values = [value for _, value in R.DEVELOPMENT]
    colors = [ORANGE] + [LIGHT] * (len(values) - 2) + [BLUE]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.62)
    for bar, value in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.25, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center",
                fontsize=8.5)
    best = min(row[1] for key, row in R.CLASSICAL["percentile"].items() if key == "fixed")
    ax.axvline(best, color=GREY, lw=0.9, ls="--")
    ax.annotate(f"classical rule using the first 32 days ({best:.2f}%)", xy=(best, 1.5), xytext=(6.5, 1.5),
                fontsize=7.8, color=GREY, va="center",
                arrowprops={"arrowstyle": "->", "color": GREY, "lw": 0.8})
    ax.set_xlim(0, 25)
    ax.set_xlabel("regret (%), lower is better")
    ax.grid(axis="y", visible=False)
    _save(fig, out_dir, "fig_development")


def information_grid(out_dir: Path) -> None:
    """Alternative design against classical rules that see the same data, and against the same KAN."""
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    series = [
        ("Classical Gamma fit*", [R.CLASSICAL["gamma"][w][1] for w in R.WINDOWS], "#BFBFBF"),
        ("Classical empirical quantile", [R.CLASSICAL["percentile"][w][1] for w in R.WINDOWS], GREY),
        ("KAN, no certificate", [R.TS_KAN[w][1] for w in R.WINDOWS], ORANGE),
        ("MonoKAN (ours)", [R.TS_MONOKAN[w][1] for w in R.WINDOWS], BLUE),
    ]
    x = np.arange(len(R.WINDOWS))
    width = 0.19
    for index, (label, values, color) in enumerate(series):
        bars = ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.2f}", ha="center", fontsize=7.2)
    ax.set_xticks(x, [R.WINDOW_LABELS[w] for w in R.WINDOWS])
    ax.set_ylabel("regret (%), lower is better")
    ax.set_ylim(0, 2.75)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=7.8)
    _save(fig, out_dir, "fig_information")


def interpretability(out_dir: Path) -> None:
    """Alternative design: what the level network learned, and how close it is to the newsvendor fractile."""
    with (INTERPRETABILITY / "partial_dependence.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (INTERPRETABILITY / "learned_level.csv").open(encoding="utf-8") as handle:
        levels = list(csv.DictReader(handle))
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    styles = {"selling price": (BLUE, "-"), "lost-sale penalty": (DARK, "-"), "purchase cost": (ORANGE, "-"),
              "holding cost": (ORANGE, "--")}
    for driver, (color, style) in styles.items():
        subset = [row for row in rows if row["driver"] == driver]
        values = np.array([float(row["feature_value"]) for row in subset])
        target = np.array([float(row["mean_target"]) for row in subset])
        position = (values - values.min()) / (values.max() - values.min())
        direction = "+" if subset[0]["certified_direction"] == "1" else "-"
        left.plot(position, target, color=color, ls=style, lw=1.7, label=f"{driver} ({direction})")
    left.set_xlabel("input, from its 1st to its 99th percentile")
    left.set_ylabel("mean target stock (units)")
    left.set_title("(a) Learned effect of each economic input")
    left.legend(fontsize=7.6, loc="best")

    learned = np.array([float(row["learned_level"]) for row in levels])
    analytic = np.array([float(row["analytic_critical_fractile"]) for row in levels])
    right.scatter(analytic, learned, s=7, color=BLUE, alpha=0.6, lw=0)
    right.plot([0, 1], [0, 1], color=GREY, lw=0.9, ls="--")
    corr = np.corrcoef(analytic, learned)[0, 1]
    right.text(0.04, 0.9, f"correlation {corr:.3f}", transform=right.transAxes, fontsize=8.5)
    right.set_xlim(0, 1)
    right.set_ylim(0, 1)
    right.set_xlabel("newsvendor fractile (p − c + k) / (p − c + k + h)")
    right.set_ylabel("learned demand percentile u")
    right.set_title("(b) Learned percentile vs newsvendor fractile")
    fig.tight_layout(w_pad=2.5)
    _save(fig, out_dir, "fig_interpretability")


def sample_efficiency(out_dir: Path) -> None:
    """Alternative design: regret against the number of training products."""
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for name, color, marker in (("MonoKAN", BLUE, "o"), ("MLP", GREY, "s")):
        ax.plot(R.SAMPLE_N, R.SAMPLE_REGRET[name], color=color, marker=marker, lw=1.6, ms=4.5,
                label=f"{name} level network")
    ax.set_xscale("log")
    ax.set_xticks(R.SAMPLE_N, [f"{n:,}" for n in R.SAMPLE_N])
    ax.minorticks_off()
    ax.set_xlabel("training products")
    ax.set_ylabel("regret (%)")
    ax.set_ylim(0, 3)
    ax.legend(fontsize=7.8)
    _save(fig, out_dir, "fig_sample_efficiency")


def make_all(out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for figure in (violations, reaction_strength, development, information_grid, interpretability,
                   sample_efficiency):
        figure(out_dir)
    return sorted(path.name for path in out_dir.glob("*.svg"))


if __name__ == "__main__":
    print(make_all(HERE / "build/figures"))
