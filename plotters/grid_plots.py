import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

def plot_pivot_table(df, threshold_low=None, threshold_high=None, cmap='bwr', figsize=(18, 30), transpose_axes=False):
    df_plot = df.copy()
    if threshold_low is not None or threshold_high is not None:
        df_plot[(df_plot < threshold_low) | (df_plot > threshold_high)] = np.nan
    
    if transpose_axes:
        df_plot = df_plot.T  # Transpose the DataFrame

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    cmap = plt.cm.get_cmap(cmap)
    cmap.set_bad('k')
    vmax = max(abs(np.nanmin(df_plot.max())), abs(np.nanmin(df_plot.min())))
    vmin = -vmax

    # Increase the aspect ratio of each cell
    #aspect_ratio = 0.1  # Adjust this value to make squares appear bigger

    plt.imshow(df_plot, cmap=cmap, vmax=vmax, vmin=vmin)  # Adjust the aspect parameter

    cm = plt.colorbar(shrink=0.3)
    plt.xticks(np.arange(len(df_plot.columns)), df_plot.columns, rotation=90, fontsize=12)
    plt.yticks(np.arange(len(df_plot.index)), df_plot.index, fontsize=12)
    axs.xaxis.set_major_locator(ticker.MultipleLocator(1))
    axs.yaxis.set_major_locator(ticker.MultipleLocator(1))
    axs.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    axs.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    axs.tick_params(axis='both', which='major', width=1.5, length=4)
    axs.grid(which='minor', color='black', linestyle='-', linewidth=1)
    plt.show()