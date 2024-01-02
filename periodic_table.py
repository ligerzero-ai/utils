import numpy as np
import pandas as pd

import os 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

from matplotlib.colors import Normalize
from pymatgen.core import Element

import warnings

module_path = os.path.dirname(os.path.abspath(__file__))
ptable = pd.read_csv(os.path.join(module_path, "periodic_table.csv"))

def get_element_number(symbol):
    try:
        return Element(symbol).Z
    except ValueError:
        warnings.warn(f"Warning: Symbol '{symbol}' was not found.")
        return np.nan
    
def get_element_symbol(element_number):
    row = ptable[ptable["Z"] == element_number]
    if not row.empty:
        return row["symbol"].values[0]
    else:
        warnings.warn(f"Warning: Element with Z:{element_number} was not found.")
        return np.nan
    
def classify_elements(element):
    # Define the properties of the different groups of elements in a dictionary
    element_groups = {
        'Actinoids': ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'],
        'Noble gases': ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og'],
        'Rare earths': ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'],
        'Transition metals': ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'],
        'Alkali metals': ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],
        'Alkaline earths': ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],
        'Halogens': ['F', 'Cl', 'Br', 'I', 'At'],
        'Metalloids': ['B', 'Si', 'Ge', 'As', 'Sb', 'Te', 'Po']
    }
    
    # Check which group the element belongs to
    for group, elements in element_groups.items():
        if element in elements:
            return group
    
    # If the element doesn't match any group, return 'Others'
    return 'Others'

def get_colour_element(element):
    # Define the color map inside the function
    color_map = {'Actinoids': 'r',
                 'Noble gases': 'royalblue',
                 'Rare earths': 'm',
                 'Transition metals': 'purple',
                 'Alkali metals': 'gold',
                 'Alkaline earths': "moccasin",
                 'Halogens': 'mediumspringgreen',
                 'Metalloids': 'darkcyan',
                 'Others': 'slategray'}

    # Classify the element using the classify_elements function
    element_group = classify_elements(element)
    
    # Assign color based on the classification using the color_map dictionary
    colour = color_map.get(element_group, 'slategray')  # Default to 'slategray' if not found in color_map
    
    return colour

def periodic_table_plot(plot_df, 
                        property = "Eseg_min",
                        count_min = None,
                        count_max = None,
                        center_cm_zero = False,
                        property_name = None,
                        cmap = cm.Blues):
    module_path = os.path.dirname(os.path.abspath(__file__))
    ptable = pd.read_csv(os.path.join(module_path, 'periodic_table.csv'))
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