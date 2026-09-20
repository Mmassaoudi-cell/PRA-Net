"""Fig. 1 (PRA-Net architecture) and Fig. 2 (source-vs-proposed conceptual
workflow) -- authored directly as simple schematic box-and-arrow diagrams,
not generated from result CSVs (see make_figures.py for the data-driven ones)."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def box(ax, x, y, w, h, text, color="#e8eef7", fontsize=9):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                        linewidth=1.2, edgecolor="#2f4f7f", facecolor=color)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.2, color="#2f4f7f")
    ax.add_patch(a)


def fig1_architecture():
    fig, ax = plt.subplots(figsize=(6.4, 7.6))
    ax.set_xlim(0, 8); ax.set_ylim(0, 12.6); ax.axis("off")

    # Row 1: input
    box(ax, 2.4, 11.2, 3.2, 1.0, "Measurement sequence $Z\\in\\mathbb{R}^{k\\times d}$")

    # Row 2: two parallel paths
    box(ax, 0.4, 9.5, 3.0, 1.0, "Per-bus feature build\n(6-dim: P/Q inj, Vmag,\nmean incident P/Q flow, degree)")
    box(ax, 4.6, 9.5, 3.0, 1.0, "DC-WLS residual\n$r=z_{dc}-H_{dc}\\hat\\theta$", color="#f7ecd8")
    arrow(ax, 3.4, 11.2, 2.2, 10.5)
    arrow(ax, 4.6, 11.2, 6.1, 10.5)

    # Row 3: merge residual into bus feature (7th dim)
    box(ax, 2.4, 7.8, 3.2, 1.0, "Residual aggregated to bus\n(concatenated as 7th feature)", color="#f7ecd8")
    arrow(ax, 1.9, 9.5, 3.0, 8.8)
    arrow(ax, 6.1, 9.5, 4.8, 8.8)

    # Row 4: GRU
    box(ax, 2.4, 6.1, 3.2, 1.0, "Per-bus GRU (shared weights)\n-- one embedding per bus --")
    arrow(ax, 4.0, 7.8, 4.0, 7.1)

    # Row 5: attention
    box(ax, 2.0, 4.3, 4.0, 1.2, "Self-attention over bus tokens\n(+ physics-residual bias $\\eta\\,\\hat\\rho\\hat\\rho^\\top$)",
        color="#e3f0e3")
    arrow(ax, 4.0, 6.1, 4.0, 5.5)

    # Row 6: two heads
    box(ax, 0.6, 2.6, 3.0, 1.0, "Dirichlet evidential head\n(detection)", color="#fbe3e3")
    box(ax, 4.4, 2.6, 3.0, 1.0, "Per-bus sigmoid head\n(localization)", color="#fbe3e3")
    arrow(ax, 3.2, 4.3, 2.4, 3.6)
    arrow(ax, 4.8, 4.3, 5.6, 3.6)

    # Row 7: outputs
    box(ax, 0.6, 0.9, 3.0, 1.0, "$p(\\text{attacked}),\\;u$\n(single-pass uncertainty)")
    box(ax, 4.4, 0.9, 3.0, 1.0, "Attacked-bus set\n(multi-label)")
    arrow(ax, 2.1, 2.6, 2.1, 1.9)
    arrow(ax, 5.9, 2.6, 5.9, 1.9)

    ax.set_title("PRA-Net: Physics-Residual Attention Network", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_architecture.png", dpi=200)
    plt.close(fig)


def fig2_workflow():
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for ax, title in zip(axes, ["Source method (reproduced)", "PRA-Net (proposed)"]):
        ax.set_xlim(0, 4); ax.set_ylim(0, 8); ax.axis("off")
        ax.set_title(title, fontsize=10)

    ax = axes[0]
    steps = ["Backbone (frozen)", "5-head ensemble\n(fine-tuned)", "LLLA per head\n(20-30 MC samples)",
             "Mutual information\n$\\to$ threshold", "VAE latent search\n(5 Adam steps/sample)",
             "Gradient attribution\n+ neighborhood filter"]
    for i, s in enumerate(reversed(steps)):
        y = 0.3 + i * 1.25
        box(ax, 0.3, y, 3.4, 1.0, s, color="#f0f0f0", fontsize=8)
        if i > 0:
            arrow(ax, 2.0, y, 2.0, y - 0.25)

    ax = axes[1]
    steps2 = ["Physics feature +\nper-bus GRU", "Residual-biased\nself-attention",
              "Evidential head\n(single pass)", "Localization head\n(single pass, joint)"]
    for i, s in enumerate(reversed(steps2)):
        y = 0.3 + i * 1.6
        box(ax, 0.3, y, 3.4, 1.2, s, color="#e3f0e3", fontsize=9)
        if i > 0:
            arrow(ax, 2.0, y, 2.0, y - 0.4)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_workflow.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1_architecture()
    fig2_workflow()
    print("done")
