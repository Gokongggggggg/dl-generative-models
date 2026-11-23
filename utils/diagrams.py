"""Report assets: schematic architecture diagrams (7 models) and mathtext equations."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch


def _box(ax, x, y, w, h, label, fc="#DCE6F1"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec="#333333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.5)


def _arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrow(x0, y, x1 - x0 - 0.02, 0, width=0.004,
                            head_width=0.035, head_length=0.02,
                            fc="#333333", ec="#333333", length_includes_head=True))


def _chain(ax, labels, y=0.5, x0=0.02, w=0.16, gap=0.045, fc="#DCE6F1"):
    x = x0
    for i, lab in enumerate(labels):
        _box(ax, x, y - 0.14, w, 0.28, lab, fc)
        if i < len(labels) - 1:
            _arrow(ax, x + w + 0.005, x + w + gap, y)
        x += w + gap
    return x


def draw_architectures(path):
    fig, axes = plt.subplots(4, 2, figsize=(11.5, 10.5))
    for ax in axes.flat:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax = axes[0, 0]
    ax.set_title("(a) MLP Classifier (Part A)", fontsize=10, loc="left")
    _chain(ax, ["input\n784", "FC+ReLU\n256", "FC+ReLU\n64 = z_MLP", "FC\n10 logits"],
           w=0.19, gap=0.05)
    ax.text(0.5, 0.12, "penultimate z_MLP diekstrak untuk Stage 2-5", ha="center", fontsize=7.5)

    ax = axes[0, 1]
    ax.set_title("(b) Vanilla Autoencoder", fontsize=10, loc="left")
    _chain(ax, ["input\n784", "Enc FC+ReLU\n256", "Enc FC+ReLU\n64 = z_AE",
                "Dec FC+ReLU\n256", "Dec FC+Tanh\n784"], w=0.155, gap=0.035)

    ax = axes[1, 0]
    ax.set_title("(c) Sparse Autoencoder", fontsize=10, loc="left")
    _chain(ax, ["input\n784", "Enc FC+ReLU\n256", "Enc FC+ReLU\n64 = h",
                "Dec FC+ReLU\n256", "Dec FC+Tanh\n784"], w=0.155, gap=0.035)
    ax.text(0.5, 0.12, "penalti L1: + lambda * ||h||_1 pada bottleneck (lambda=1e-4)",
            ha="center", fontsize=7.5, color="#8B0000")

    ax = axes[1, 1]
    ax.set_title("(d) VAE", fontsize=10, loc="left")
    _chain(ax, ["input\n784", "FC+ReLU\n256"], w=0.16, gap=0.04)
    _box(ax, 0.42, 0.62, 0.16, 0.22, "mu\n64")
    _box(ax, 0.42, 0.16, 0.16, 0.22, "logvar\n64")
    _arrow(ax, 0.385, 0.42, 0.73); _arrow(ax, 0.385, 0.42, 0.27)
    _box(ax, 0.63, 0.39, 0.15, 0.22, "reparam\nz=mu+sigma*eps", fc="#FDE9D9")
    _arrow(ax, 0.585, 0.63, 0.73); _arrow(ax, 0.585, 0.63, 0.27)
    _box(ax, 0.82, 0.62, 0.15, 0.22, "Dec 256\nTanh 784")
    _arrow(ax, 0.785, 0.82, 0.5)
    ax.text(0.5, 0.05, "KL(q(z|x) || N(0,I)) membentuk prior yang dapat disampling",
            ha="center", fontsize=7.5)

    ax = axes[2, 0]
    ax.set_title("(e) Vanilla GAN (MLP)", fontsize=10, loc="left")
    ax.text(0.02, 0.82, "G:", fontsize=8)
    _chain(ax, ["z\n100", "FC+LReLU\n256", "FC+LReLU\n512", "FC+Tanh\n784"],
           y=0.72, w=0.18, gap=0.045)
    ax.text(0.02, 0.36, "D:", fontsize=8)
    _chain(ax, ["x\n784", "FC+LReLU\n512", "FC+LReLU\n256", "FC\n1 logit"],
           y=0.26, w=0.18, gap=0.045, fc="#E2EFDA")

    ax = axes[2, 1]
    ax.set_title("(f) DCGAN (28x28)", fontsize=10, loc="left")
    ax.text(0.02, 0.82, "G:", fontsize=8)
    _chain(ax, ["z 100\nproj", "ConvT+BN\n256x7x7", "ConvT+BN\n128x14x14",
                "ConvT+BN\n64x28x28", "Conv+Tanh\n1x28x28"],
           y=0.72, w=0.155, gap=0.032)
    ax.text(0.02, 0.36, "D:", fontsize=8)
    _chain(ax, ["1x28x28", "Conv+LReLU\n64x14x14", "Conv+BN\n128x7x7",
                "Conv+BN\n256x7x7", "FC\n1 logit"],
           y=0.26, w=0.155, gap=0.032, fc="#E2EFDA")

    ax = axes[3, 0]
    ax.set_title("(g) DDPM U-Net (eps-prediction)", fontsize=10, loc="left")
    _chain(ax, ["x_t + t-emb\n1x28x28", "ResBlock\n64x28x28", "down\n64x14x14",
                "ResBlock\n128x7x7", "mid\n128x7x7"], y=0.68, w=0.155, gap=0.032)
    _chain(ax, ["up+skip\n64x14x14", "up+skip\n64x28x28", "out conv\neps 1x28x28"],
           y=0.28, x0=0.2, w=0.19, gap=0.06, fc="#FDE9D9")
    ax.annotate("", xy=(0.28, 0.42), xytext=(0.28, 0.54),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0.62, 0.45, "skip connections 28 & 14;\nsinusoidal time embedding -> MLP -> ResBlocks",
            fontsize=7.5, va="center")

    ax = axes[3, 1]
    ax.axis("off")
    ax.text(0.02, 0.85, "Konvensi diagram:", fontsize=9, weight="bold")
    ax.text(0.02, 0.68, "Biru = jalur encoder / generator;\nhijau = discriminator;\n"
            "oranye = operasi stochastic / decoding.", fontsize=8, va="top")
    ax.text(0.02, 0.35, "Semua model berbagi split 80/10/10 (seed 42),\n"
            "normalisasi [-1,1], batch 128, optimizer Adam.", fontsize=8, va="top")

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_equation(tex, path, fontsize=12):
    fig = plt.figure(figsize=(8, 0.9))
    fig.text(0.02, 0.5, tex, fontsize=fontsize, va="center")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
