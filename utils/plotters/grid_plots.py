import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_pivot_table(
    df,
    colormap_thresholds=[None, None],
    figsize=(18, 30),
    colormap="bwr",
    colormap_label="E$_{\\rm{seg}}$ (eV)",
    color_label_fontsize=20,
    colormap_tick_fontsize=12,
    xtick_fontsize=18,
    ytick_fontsize=12,
    threshold_low=None,
    threshold_high=None,
    transpose_axes=False,
):
    """
    Plot a heatmap with custom parameters.

    Parameters:
    - df: DataFrame to plot.
    - colormap_thresholds: List with [vmin, vmax] for the colormap, default is [-1, 1].
    - figsize: Tuple for figure size (width, height), default is (18, 30).
    - colormap: String for the colormap name, default is 'bwr'.
    - colormap_label: Label for the colorbar, default is 'E$_{\\rm{seg}}$ (eV)'.
    - fontsize: Font size for the colorbar label, default is 20.
    - xtick_fontsize: Font size for x-axis tick labels, default is 18.
    - ytick_fontsize: Font size for y-axis tick labels, default is 12.
    - threshold_low: Lower threshold to filter data; defaults to None.
    - threshold_high: Higher threshold to filter data; defaults to None.
    - transpose_axes: Boolean to transpose the DataFrame; defaults to False.
    """
    if threshold_low is not None or threshold_high is not None:
        df = df.copy()
        df[(df < threshold_low) | (df > threshold_high)] = np.nan

    if transpose_axes:
        df = df.T

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    cmap = plt.get_cmap(colormap)
    cmap.set_bad("k")
    if colormap_thresholds == [None, None]:
        vmax = max(abs(np.nanmin(df.max())), abs(np.nanmin(df.min())))
        vmin = -vmax
    else:
        vmin, vmax = colormap_thresholds
    im = axs.imshow(df, cmap=cmap, vmax=vmax, vmin=vmin)
    cm = plt.colorbar(im, ax=axs, shrink=0.3, location="right", pad=0.01)
    cm.set_label(
        colormap_label, rotation=270, labelpad=15, fontsize=color_label_fontsize
    )
    # cm.ax.tick_params(labelsize=colormap_tick_fontsize)  # Set colorbar tick label size
    if colormap_thresholds != [None, None]:
        ticks = cm.get_ticks()
        if len(ticks) > 1:  # Check to ensure there are ticks to modify
            tick_labels = [
                (
                    f"$<{vmin}$"
                    if i == 0
                    else f"$>{vmax}$" if i == len(ticks) - 1 else str(tick)
                )
                for i, tick in enumerate(ticks)
            ]
            cm.set_ticks(ticks)  # Set the ticks back if they were changed
            cm.set_ticklabels(
                tick_labels, fontsize=colormap_tick_fontsize
            )  # Set the modified tick labels
    else:
        cm.set_ticklabels(
            cm.get_ticks(), fontsize=colormap_tick_fontsize
        )  # Set the modified tick labels

    plt.xticks(
        np.arange(len(df.columns)), df.columns, rotation=0, fontsize=xtick_fontsize
    )
    plt.yticks(np.arange(len(df.index)), df.index, fontsize=ytick_fontsize)

    axs.xaxis.set_major_locator(ticker.MultipleLocator(1))
    axs.yaxis.set_major_locator(ticker.MultipleLocator(1))
    axs.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    axs.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    axs.tick_params(axis="both", which="major", width=1.5, length=4)
    axs.grid(which="minor", color="black", linestyle="-", linewidth=1)

    return fig, axs
