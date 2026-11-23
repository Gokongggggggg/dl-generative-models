"""Figure helpers (300 DPI PNG output)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from configs import config
from utils.dataset import denorm


def to_img(x):
    """(1,28,28) or (784,) tensor in [-1,1] -> 28x28 numpy in [0,1]."""
    return denorm(x.detach().cpu()).reshape(28, 28).numpy()


def panel_grid(fig, spec, images, nrows, ncols, title, row_labels=None):
    inner = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=spec,
                                    hspace=0.08, wspace=0.05)
    for i in range(nrows * ncols):
        ax = fig.add_subplot(inner[i // ncols, i % ncols])
        ax.imshow(to_img(images[i]), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if row_labels is not None and i % ncols == 0:
            ax.set_ylabel(row_labels[i // ncols], fontsize=9, rotation=90, labelpad=8)
    # panel title above the grid
    ax0 = fig.add_subplot(spec, frame_on=False)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_title(title, fontsize=12, pad=10)


def save_fig(fig, filename):
    path = config.FIG_DIR / filename
    # nested-gridspec ylabels can fall outside the automatic tight bbox
    extra = [ax.yaxis.label for ax in fig.axes if ax.yaxis.label.get_text()]
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight", bbox_extra_artists=extra)
    plt.close(fig)
    print(f"Saved {path}")
