import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

from matplotlib.colors import Normalize

element_symbols = {
    1: 'H',
    2: 'He',
    3: 'Li',
    4: 'Be',
    5: 'B',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    10: 'Ne',
    11: 'Na',
    12: 'Mg',
    13: 'Al',
    14: 'Si',
    15: 'P',
    16: 'S',
    17: 'Cl',
    18: 'Ar',
    19: 'K',
    20: 'Ca',
    21: 'Sc',
    22: 'Ti',
    23: 'V',
    24: 'Cr',
    25: 'Mn',
    26: 'Fe',
    27: 'Co',
    28: 'Ni',
    29: 'Cu',
    30: 'Zn',
    31: 'Ga',
    32: 'Ge',
    33: 'As',
    34: 'Se',
    35: 'Br',
    36: 'Kr',
    37: 'Rb',
    38: 'Sr',
    39: 'Y',
    40: 'Zr',
    41: 'Nb',
    42: 'Mo',
    43: 'Tc',
    44: 'Ru',
    45: 'Rh',
    46: 'Pd',
    47: 'Ag',
    48: 'Cd',
    49: 'In',
    50: 'Sn',
    51: 'Sb',
    52: 'Te',
    53: 'I',
    54: 'Xe',
    55: 'Cs',
    56: 'Ba',
    57: 'La',
    58: 'Ce',
    59: 'Pr',
    60: 'Nd',
    61: 'Pm',
    62: 'Sm',
    63: 'Eu',
    64: 'Gd',
    65: 'Tb',
    66: 'Dy',
    67: 'Ho',
    68: 'Er',
    69: 'Tm',
    70: 'Yb',
    71: 'Lu',
    72: 'Hf',
    73: 'Ta',
    74: 'W',
    75: 'Re',
    76: 'Os',
    77: 'Ir',
    78: 'Pt',
    79: 'Au',
    80: 'Hg',
    81: 'Tl',
    82: 'Pb',
    83: 'Bi',
    84: 'Po',
    85: 'At',
    86: 'Rn',
    87: 'Fr',
    88: 'Ra',
    89: 'Ac',
    90: 'Th',
    91: 'Pa',
    92: 'U'}

def get_element_number(symbol):
    for number, sym in reversed(element_symbols.items()):
        if sym == symbol:
            return number
    print(f"Warning: Symbol {symbol} was not found in the dictionary.")
    return np.nan

def get_element_symbol(element_number):
    try:
        return element_symbols[element_number]
    except KeyError:
        print(f"Warning: Element symbol for element number {element_number} was not found in the dictionary.")
        return np.nan

def periodic_table_plot(plot_df, 
                        property = "Eseg_min",
                        count_min = None,
                        count_max = None,
                        center_cm_zero = False,
                        property_name = None,
                        cmap = cm.Blues):
    module_path = os.path.dirname(os.path.abspath(__file__))
    ptable = pd.read_csv(os.path.join(module_path, 'periodic_table.csv'))s
    ptable.index = ptable['symbol'].values
    elem_tracker = ptable['count']
    n_row = ptable['row'].max()
    n_column = ptable['column'].max()

    fig, ax = plt.subplots(figsize=(n_column, n_row))
    rows = ptable['row']
    columns = ptable['column']
    symbols = ptable['symbol']
    rw = 0.9  # rectangle width (rw)
    rh = rw  # rectangle height (rh)
    for row, column, symbol in zip(rows, columns, symbols):
        row = ptable['row'].max() - row

        if count_min == None:
            count_min = plot_df[property].min() 
        if count_max == None:
            count_max = plot_df[property].max()
        # Normalise around 0
        if center_cm_zero:
            cm_threshold = max(abs(count_min), abs(count_max))
            norm = Normalize(-cm_threshold, cm_threshold)
        else:
            norm = Normalize(vmin=count_min, vmax=count_max)
        count = elem_tracker[symbol]
        if symbol in plot_df.element.unique():
            count = plot_df[plot_df["element"] == symbol][property].values[0]
        else:
            count = 0
        # if log_scale:
        #     norm = Normalize(vmin=np.log(1), vmax=np.log(count_max))
        #     if count != 0:
        #         count = np.log(count)
        color = cmap(norm(count))
        if count == 0:
            color = 'silver'
        if row < 3:
            row += 0.5
        rect = patches.Rectangle((column, row), rw, rh,
                                    linewidth=1.5,
                                    edgecolor='gray',
                                    facecolor=color,
                                    alpha=1)

        plt.text(column+rw/2, row+rw/2, symbol,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    fontweight='semibold', color='k')

        ax.add_patch(rect)

    granularity = 20

    if center_cm_zero:
        colormap_array = np.linspace(-cm_threshold, cm_threshold, granularity)
    else:
        colormap_array = range(granularity)
        
    for i, value in enumerate(colormap_array):
        if center_cm_zero:
            value = value
        else:
            value = (i) * count_max/(granularity-1)
        # if log_scale:
        #     if value != 0:
        #         value = np.log(value)
        color = cmap(norm(value))
        if value == 0:
            color = 'silver'
        length = 9
        x_offset = 3.5
        y_offset = 7.8
        x_loc = i/(granularity) * length + x_offset
        width = length / granularity
        height = 0.35
        rect = patches.Rectangle((x_loc, y_offset), width, height,
                                    linewidth=1.5,
                                    edgecolor='gray',
                                    facecolor=color,
                                    alpha=1)

        if i in [0, 4, 9, 14, 19]:
            text = f'{value:.1f}'
            if center_cm_zero:
                if i == 9:
                    text = "0"
            # if log_scale:
            #     text = f'{np.exp(value):0.1e}'.replace('+', '')
            plt.text(x_loc+width/2, y_offset-0.4, text,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontweight='semibold',
                        fontsize=20, color='k')

        ax.add_patch(rect)
    if property_name == None:
        property_name = property
    plt.text(x_offset+length/2, y_offset+0.7,
                property_name,
                horizontalalignment='center',
                verticalalignment='center',
                fontweight='semibold',
                fontsize=20, color='k')
    ax.set_ylim(-0.15, n_row+.1)
    ax.set_xlim(0.85, n_column+1.1)

    # fig.patch.set_visible(False)
    ax.axis('off')

    plt.draw()
    plt.pause(0.001)
    plt.close()