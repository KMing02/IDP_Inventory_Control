"""Generate a sectioned PDF report for the IDP monotonic inventory benchmark."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "outputs" / "idp_benchmark" / "structure-ls_compact_validation"
SAMPLE_DIR = ROOT / "outputs" / "idp_benchmark" / "sample-efficiency_compact_validation"
PLOT_DIR = ROOT / "outputs" / "notebook_plots"
OUT_DIR = ROOT / "output" / "pdf"
PREVIEW_DIR = ROOT / "tmp" / "pdfs"


PAGE_SIZE = (8.5, 11.0)
FONT = "DejaVu Sans"


def _ensure_inputs() -> None:
    required = [
        ARTIFACT_DIR / "summary.csv",
        ARTIFACT_DIR / "paper_table3_replication.csv",
        ARTIFACT_DIR / "training_log.csv",
        SAMPLE_DIR / "sample_efficiency.csv",
        PLOT_DIR / "dashboard_better_faster_interpretable.png",
        PLOT_DIR / "reward_structure_tradeoff.png",
        PLOT_DIR / "policy_curves.png",
        PLOT_DIR / "order_up_to_diagnostics.png",
        PLOT_DIR / "training_curves.png",
        PLOT_DIR / "sample_efficiency.png",
        PLOT_DIR / "monokan_splines.png",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing report inputs: " + ", ".join(missing))


def _fmt(value: float | int | str | None, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (float, int)):
        if math.isnan(float(value)):
            return ""
        if abs(float(value)) >= 100:
            return f"{float(value):,.1f}"
        return f"{float(value):,.{digits}f}"
    return str(value)


def _page(title: str, subtitle: str | None = None):
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    fig.text(0.07, 0.955, title, ha="left", va="top", fontsize=18, fontweight="bold", family=FONT)
    if subtitle:
        fig.text(0.07, 0.922, subtitle, ha="left", va="top", fontsize=9.5, color="#555555", family=FONT)
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.895, 0.895], color="#333333", linewidth=0.8))
    return fig


def _wrapped(fig, x: float, y: float, text: str, width: int = 92, size: float = 10.0, color: str = "#222222") -> float:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=width))
    step = 0.020 * (size / 10.0)
    for line in lines:
        fig.text(x, y, line, ha="left", va="top", fontsize=size, color=color, family=FONT)
        y -= step
    return y


def _bullets(fig, x: float, y: float, items: list[str], width: int = 88, size: float = 10.0) -> float:
    for item in items:
        lines = textwrap.wrap(item, width=width)
        if not lines:
            continue
        fig.text(x, y, "- " + lines[0], ha="left", va="top", fontsize=size, family=FONT)
        y -= 0.020 * (size / 10.0)
        for line in lines[1:]:
            fig.text(x + 0.018, y, line, ha="left", va="top", fontsize=size, family=FONT)
            y -= 0.020 * (size / 10.0)
        y -= 0.006
    return y


def _table(fig, rect, df: pd.DataFrame, fontsize: float = 8.5, scale_y: float = 1.25) -> None:
    ax = fig.add_axes(rect)
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=list(df.columns),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, scale_y)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor("#F0F3F6")
            cell.set_text_props(weight="bold", color="#111111")
        else:
            cell.set_facecolor("white")


def _image(fig, path: Path, rect, title: str | None = None) -> None:
    ax = fig.add_axes(rect)
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, pad=4, family=FONT)


def _footer(fig, page_no: int) -> None:
    fig.text(0.07, 0.035, "IDP Monotonic Inventory Benchmark - compact validation artifacts", fontsize=8, color="#777777")
    fig.text(0.93, 0.035, str(page_no), ha="right", fontsize=8, color="#777777")


def _add_page(pdf: PdfPages, fig, page_no: int) -> None:
    _footer(fig, page_no)
    pdf.savefig(fig, bbox_inches="tight")
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW_DIR / f"idp_report_page_{page_no:02d}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _contact_sheet(page_count: int) -> Path:
    images = [Image.open(PREVIEW_DIR / f"idp_report_page_{idx:02d}.png").convert("RGB") for idx in range(1, page_count + 1)]
    thumb_w = 340
    thumbs = []
    for image in images:
        ratio = thumb_w / image.width
        thumbs.append(image.resize((thumb_w, int(image.height * ratio))))
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    margin = 24
    label_h = 22
    cell_h = max(t.height for t in thumbs) + label_h
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * margin, rows * cell_h + (rows + 1) * margin), "white")
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = margin + col * (thumb_w + margin)
        y = margin + row * (cell_h + margin)
        sheet.paste(thumb, (x, y + label_h))
    out = PREVIEW_DIR / "idp_report_contact_sheet.png"
    sheet.save(out)
    return out


def main() -> None:
    _ensure_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(ARTIFACT_DIR / "summary.csv")
    table3 = pd.read_csv(ARTIFACT_DIR / "paper_table3_replication.csv")
    sample_eff = pd.read_csv(SAMPLE_DIR / "sample_efficiency.csv")

    indexed = summary.set_index("policy_name")
    fcnn = indexed.loc["PaperFCNN"]
    mono = indexed.loc["MonoKAN"]
    penalty = indexed.loc["PenaltyMLP"]
    reward_gap = 100.0 * (mono["mean_reward_per_period"] / fcnn["mean_reward_per_period"] - 1.0)
    speedup = fcnn["training_seconds"] / mono["training_seconds"] if mono["training_seconds"] > 0 else float("nan")
    accepted_reward = reward_gap >= -0.5
    accepted_structure = mono["raw_violation_rate"] == 0 and mono["deployed_violation_rate"] == 0

    pdf_path = OUT_DIR / "idp_monotonic_inventory_report.pdf"
    page = 1
    with PdfPages(pdf_path) as pdf:
        fig = _page("Monotonic Architectures for Lost-Sales Inventory Control", "Section 1 - project objective and current compact result")
        y = 0.86
        y = _wrapped(
            fig,
            0.07,
            y,
            "Project description: replace loss-function penalties for monotonicity with architecturally enforced constraints, then test whether the resulting policy is better structured, faster to train, and more interpretable than penalty-based or unconstrained networks.",
            width=96,
            size=10.5,
        )
        y -= 0.02
        status = [
            f"Structure criterion: MonoKAN raw and deployed monotonicity violations are {_fmt(mono['raw_violation_rate'])} and {_fmt(mono['deployed_violation_rate'])}; this passes the architecture-enforced monotonicity check.",
            f"Performance criterion: MonoKAN reward is {_fmt(mono['mean_reward_per_period'])} per period versus PaperFCNN {_fmt(fcnn['mean_reward_per_period'])}, a {reward_gap:.1f}% gap; this compact run does not yet meet the match-FCNN requirement.",
            f"Efficiency criterion: compact training time is {_fmt(mono['training_seconds'])} seconds for MonoKAN versus {_fmt(fcnn['training_seconds'])} seconds for PaperFCNN, about {_fmt(speedup, 1)}x faster in this smoke setting.",
            "Interpretability criterion: MonoKAN exposes monotone spline components and exact zero violations; the FCNN and penalty models can appear nonincreasing on the sampled grid but do not encode the order-up-to slope by construction.",
        ]
        y = _bullets(fig, 0.09, y, status, width=92, size=10)
        y -= 0.01
        _wrapped(
            fig,
            0.07,
            y,
            "Important caveat: these numbers are from compact validation artifacts. They verify the pipeline and reveal model behavior, but they are not final paper-replication evidence. The full paper_replication profile is needed for scientific comparison.",
            width=96,
            size=9.5,
            color="#7A3B00",
        )
        decision = pd.DataFrame(
            [
                ["Architecture-enforced monotonicity", "Pass" if accepted_structure else "Fail"],
                ["Reward no worse than FCNN by 0.5%", "Pass" if accepted_reward else "Fail"],
                ["Primary next action", "Tune MonoKAN capacity/training under LS before extensions"],
            ],
            columns=["Acceptance item", "Current compact result"],
        )
        _table(fig, [0.11, 0.13, 0.78, 0.18], decision, fontsize=9.5, scale_y=1.5)
        _add_page(pdf, fig, page)
        page += 1

        fig = _page("Mathematical Formulation", "Section 2 - lost-sales dynamics, base-stock structure, and diagnostics")
        y = 0.855
        equations = (
            "State y_t is on-hand inventory and q_t is the purchase quantity. With no lead time and lost sales, available inventory is a_t = y_t + q_t, sales are min(d_t, a_t), and next inventory is y_{t+1} = max(a_t - d_t, 0). The one-period reward used for evaluation is revenue minus purchase and holding costs.\n"
            "For a Gamma-demand product, the optimal no-lead-time base-stock policy has q*(y) = max(S - y, 0), where S is the critical-quantile order-up-to level. Therefore the desired structural derivative is dq*/dy = -1 while y < S, and dq*/dy = 0 after the cap at zero order.\n"
            "Monotonicity means y_2 > y_1 implies q_theta(y_2, z) <= q_theta(y_1, z), where z contains product and demand-history features. A penalty baseline adds loss terms for violations. MonoKAN instead restricts the function class so this inequality is enforced by the architecture."
        )
        y = _wrapped(fig, 0.07, y, equations, width=98, size=10)
        y -= 0.025
        metrics = pd.DataFrame(
            [
                ["Reward gap vs FCNN", "100 * (R_model / R_FCNN - 1)", "Performance constraint"],
                ["Raw violation rate", "mean[ max(q(y_hi)-q(y_lo), 0) > tol ]", "Monotonicity before deployment caps"],
                ["Deployed violation rate", "same check after action cap", "Operational monotonicity"],
                ["Slope deviation", "mean |Delta q / Delta y + 1| in active region", "Order-up-to shape"],
                ["Order-up-to error", "|(q(0)) - S_oracle| or grid analogue", "Structural diagnostic only"],
            ],
            columns=["Metric", "Definition", "Interpretation"],
        )
        _table(fig, [0.06, 0.26, 0.88, 0.35], metrics, fontsize=8.2, scale_y=1.65)
        _wrapped(
            fig,
            0.07,
            0.19,
            "Graphically, a good policy curve decreases linearly with slope -1 until it reaches zero order. A monotone curve with the wrong slope can still pass violation tests while failing the order-up-to structural test.",
            width=96,
            size=10,
        )
        _add_page(pdf, fig, page)
        page += 1

        fig = _page("Operational Results", "Section 3 - compact reward and service metrics")
        op = summary[
            [
                "policy_name",
                "mean_reward_per_period",
                "service_level",
                "lost_sales_rate",
                "mean_order_quantity",
                "mean_inventory",
                "training_seconds",
            ]
        ].copy()
        op.columns = ["Policy", "Reward/period", "Service", "Lost sales", "Mean order", "Mean inventory", "Train sec"]
        for col in op.columns[1:]:
            op[col] = op[col].map(lambda value: _fmt(value))
        _table(fig, [0.05, 0.49, 0.90, 0.30], op, fontsize=7.5, scale_y=1.35)
        table3_fmt = table3.copy()
        table3_fmt.columns = ["Policy", "Average reward", "Gap to omniscient (%)"]
        table3_fmt["Average reward"] = table3_fmt["Average reward"].map(lambda value: _fmt(value))
        table3_fmt["Gap to omniscient (%)"] = table3_fmt["Gap to omniscient (%)"].map(lambda value: _fmt(value))
        _table(fig, [0.12, 0.16, 0.76, 0.25], table3_fmt, fontsize=8.2, scale_y=1.35)
        _wrapped(
            fig,
            0.07,
            0.855,
            "Interpretation: in the compact run, PaperFCNN and PenaltyMLP tie on reward because they share the same initialization and the penalty has no sampled violations. MonoKAN is structurally cleaner but over-orders, creating high inventory and a large reward gap. The compact Table-3-style ranking is not paper-faithful because the product count and horizon are intentionally tiny.",
            width=96,
            size=10,
        )
        _add_page(pdf, fig, page)
        page += 1

        fig = _page("Structure Dashboard", "Section 4 - better, faster, interpretable criteria")
        _image(fig, PLOT_DIR / "dashboard_better_faster_interpretable.png", [0.06, 0.36, 0.88, 0.47])
        _wrapped(
            fig,
            0.07,
            0.30,
            f"Mathematical reading: MonoKAN has zero violation rate and zero slope deviation on the diagnostic grid, so it satisfies q(y_hi) <= q(y_lo) exactly in the tested raw and deployed policies. However, the reward gap is {reward_gap:.1f}%, so structure is currently bought at unacceptable reward loss.",
            width=96,
            size=10,
        )
        _add_page(pdf, fig, page)
        page += 1

        fig = _page("Reward-Structure Trade-off", "Section 5 - why the project is not just a reward benchmark")
        _image(fig, PLOT_DIR / "reward_structure_tradeoff.png", [0.08, 0.38, 0.84, 0.45])
        _wrapped(
            fig,
            0.07,
            0.31,
            "Graphical reading: the desired point is near zero violations and near zero reward gap to PaperFCNN. PenaltyMLP sits on the FCNN performance point in this compact run but does not add architecture-level guarantees. MonoKAN sits at the structural target but not the reward target.",
            width=96,
            size=10,
        )
        _add_page(pdf, fig, page)
        page += 1

        fig = _page("Policy Curves", "Section 6 - purchase quantity as a function of inventory")
        _image(fig, PLOT_DIR / "policy_curves.png", [0.06, 0.30, 0.88, 0.55])
        _wrapped(
            fig,
            0.07,
            0.245,
            "Mathematical reading: for an order-up-to policy, each curve should look like q(y)=max(S-y,0). The important visual signature is a straight line with slope -1 until it hits zero. Smoothness alone is not enough; the curve must have the correct economic slope.",
            width=96,
            size=10,
        )
        _add_page(pdf, fig, page)
        page += 1

        fig = _page("Order-Up-To Diagnostics", "Section 7 - structural agreement with Gamma critical quantiles")
        _image(fig, PLOT_DIR / "order_up_to_diagnostics.png", [0.06, 0.34, 0.88, 0.50])
        _wrapped(
            fig,
            0.07,
            0.28,
            "Interpretation: the oracle base-stock level S comes from the exact Gamma critical quantile. Order-up-to error is not used as a supervised training objective; it is an ex-post structural diagnostic showing whether the learned policy recovered the economic form of the solution.",
            width=96,
            size=10,
        )
        _add_page(pdf, fig, page)
        page += 1

        fig = _page("Training And Sample Efficiency", "Section 8 - exploratory efficiency evidence")
        _image(fig, PLOT_DIR / "training_curves.png", [0.06, 0.50, 0.88, 0.32])
        _image(fig, PLOT_DIR / "sample_efficiency.png", [0.09, 0.20, 0.82, 0.23])
        sample_fmt = sample_eff.copy()
        sample_fmt.columns = ["H", "N", "Policy", "Mean total reward", "Train sec"]
        sample_fmt["Mean total reward"] = sample_fmt["Mean total reward"].map(lambda value: _fmt(value))
        sample_fmt["Train sec"] = sample_fmt["Train sec"].map(lambda value: _fmt(value))
        _table(fig, [0.21, 0.055, 0.58, 0.09], sample_fmt, fontsize=7.5, scale_y=1.15)
        _wrapped(
            fig,
            0.07,
            0.455,
            "Interpretation: sample efficiency is an open research question. The current compact grid is a smoke test only, so it should not be used to claim faster learning. A credible conclusion requires the paper-aligned N and H grid with both FCNN and MonoKAN.",
            width=96,
            size=9.5,
        )
        _add_page(pdf, fig, page)
        page += 1

        fig = _page("MonoKAN Spline Interpretability", "Section 9 - architecture-level monotone components")
        _image(fig, PLOT_DIR / "monokan_splines.png", [0.06, 0.33, 0.88, 0.50])
        _wrapped(
            fig,
            0.07,
            0.27,
            "Graphical reading: the first-layer inventory splines expose how the inventory coordinate enters the monotone network. This is the main interpretability difference from an unconstrained FCNN: the shape restriction can be inspected directly rather than inferred from penalty terms after training.",
            width=96,
            size=10,
        )
        _add_page(pdf, fig, page)
        page += 1

        fig = _page("Conclusions And Next Steps", "Section 10 - project-aligned interpretation")
        y = 0.85
        conclusions = [
            "The current implementation matches the project direction: LS-first, synthesized products, paper-style baselines, architecture-enforced monotonicity, penalty kept as a reference baseline.",
            "The compact evidence supports the structural claim for MonoKAN: zero raw and deployed violations, plus interpretable splines.",
            "The compact evidence does not yet support the full project success claim because MonoKAN does not match FCNN reward. This must be fixed before claiming an advantage.",
            "The penalty baseline is not the proposed method. It remains useful because it mirrors the reference paper's regularization idea and demonstrates why architectural enforcement is a sharper structural intervention.",
        ]
        y = _bullets(fig, 0.09, y, conclusions, width=90, size=10)
        y -= 0.015
        next_steps = pd.DataFrame(
            [
                ["1", "Run paper_replication LS baseline", "Establish credible FCNN and optimal baselines"],
                ["2", "Tune MonoKAN under identical LS data", "Close reward gap without adding monotonicity penalties"],
                ["3", "Run structure-ls full validation", "Produce final policy curves and violation diagnostics"],
                ["4", "Run protected final test once", "Report held-out result only after model choices are frozen"],
            ],
            columns=["Step", "Action", "Reason"],
        )
        _table(fig, [0.06, 0.22, 0.88, 0.30], next_steps, fontsize=8.5, scale_y=1.55)
        _wrapped(
            fig,
            0.07,
            0.15,
            "Bottom line: the report shows mathematically and graphically that architectural monotonicity is working, but the current compact MonoKAN policy is not yet operationally competitive with the FCNN.",
            width=96,
            size=10.5,
        )
        _add_page(pdf, fig, page)

    contact = _contact_sheet(page)
    print(f"Wrote {pdf_path.relative_to(ROOT)}")
    print(f"Wrote {contact.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
