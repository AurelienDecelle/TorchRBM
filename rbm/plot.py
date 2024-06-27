import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns


def plot_scatter_labels(ax, data_proj, gen_data_proj, proj1, proj2, labels):
    ax.scatter(
        data_proj[:, proj1],
        data_proj[:, proj2],
        color="black",
        s=50,
        label=labels[0],
        zorder=0,
        alpha=0.3,
    )
    ax.scatter(
        gen_data_proj[:, proj1],
        gen_data_proj[:, proj2],
        color="red",
        label=labels[1],
        s=20,
        zorder=2,
        edgecolor="black",
        marker="o",
        alpha=1,
        linewidth=0.4,
    )


def plot_hist(
    ax, data_proj, gen_data_proj, color, proj, labels, orientation="vertical"
):
    ax.hist(
        data_proj[:, proj],
        bins=40,
        color="black",
        histtype="step",
        label=labels[0],
        zorder=0,
        density=True,
        orientation=orientation,
        lw=1,
    )
    ax.hist(
        gen_data_proj[:, proj],
        bins=40,
        color=color,
        histtype="step",
        label=labels[1],
        zorder=1,
        density=True,
        orientation=orientation,
        lw=1.5,
    )
    ax.axis("off")


def plot_PCA(data1, data2, labels, dir1=0, dir2=1):
    fig = plt.figure(dpi=100, figsize=(5, 5))
    gs = GridSpec(4, 4)

    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_hist_x = fig.add_subplot(gs[0, 0:3])
    ax_hist_y = fig.add_subplot(gs[1:4, 3])

    plot_scatter_labels(ax_scatter, data1, data2, dir1, dir2, labels=labels)
    plot_hist(ax_hist_x, data1, data2, "red", dir1, labels=labels)
    plot_hist(
        ax_hist_y, data1, data2, "red", dir2, orientation="horizontal", labels=labels
    )

    ax_hist_x.legend(fontsize=12, bbox_to_anchor=(1, 1))
    h, l = ax_scatter.get_legend_handles_labels()
    ax_scatter.set_xlabel(f"PC{dir1}")
    ax_scatter.set_ylabel(f"PC{dir2}")