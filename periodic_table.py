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
        'Metalloids': ['B', 'Si', 'Ge', 'As', 'Sb', 'Te', 'Po'],
        'Reactive nonmetals': ['H', 'C', 'N', 'O', 'P', 'S', 'Se'],  # Excluding Halogens as they're classified separately
        'Post-transition metals': ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi']
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
                        property="Eseg_min",
                        count_min=None,
                        count_max=None,
                        center_cm_zero=False,
                        center_point=None,  # New parameter for arbitrary centering
                        property_name=None,
                        cmap=cm.Blues,
                        element_font_color = "darkgoldenrod"
):
    module_path = os.path.dirname(os.path.abspath(__file__))
    ptable = pd.read_csv(os.path.join(module_path, 'periodic_table.csv'))
    ptable.index = ptable['symbol'].values
    elem_tracker = ptable['count']
    ptable = ptable[ptable['Z'] <= 92]  # Cap at element 92

    n_row = ptable['row'].max()
    n_column = ptable['column'].max()

    fig, ax = plt.subplots(figsize=(n_column, n_row))
    rows = ptable['row']
    columns = ptable['column']
    symbols = ptable['symbol']
    rw = 0.9  # rectangle width
    rh = rw    # rectangle height

    if count_min is None:
        count_min = plot_df[property].min()
    if count_max is None:
        count_max = plot_df[property].max()

    # Adjust normalization based on centering preference
    if center_cm_zero:
        cm_threshold = max(abs(count_min), abs(count_max))
        norm = Normalize(-cm_threshold, cm_threshold)
    elif center_point is not None:
        # Adjust normalization to center around the arbitrary point
        max_diff = max(center_point - count_min, count_max - center_point)
        norm = Normalize(center_point - max_diff, center_point + max_diff)
    else:
        norm = Normalize(vmin=count_min, vmax=count_max)

    for row, column, symbol in zip(rows, columns, symbols):
        row = ptable['row'].max() - row
        if symbol in plot_df.element.unique():
            count = plot_df[plot_df["element"] == symbol][property].values[0]
            # Check for NaN and adjust color and skip text accordingly
            if pd.isna(count):
                color = 'grey'  # Set color to none for NaN values
                count = ''  # Avoid displaying text for NaN values
            else:
                color = cmap(norm(count))
        else:
            count = ''
            color = 'none'

        if row < 3:
            row += 0.5
        rect = patches.Rectangle((column, row), rw, rh,
                                linewidth=1.5,
                                edgecolor='gray',
                                facecolor=color,
                                alpha=1)

        # Element symbol
        plt.text(column + rw / 2, row + rh / 2 + 0.2, symbol,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=22,  # Adjusted for visibility
                fontweight='semibold',
                color=element_font_color)

        # Property value - Added below the symbol
        if count:  # Only display if count is not empty (including not NaN)
            plt.text(column + rw / 2, row + rh / 2 - 0.25, f"{count:.2f}",  # Formatting count to 2 decimal places
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,  # Smaller font size for the count value
                    fontweight='semibold',
                    color=element_font_color)

        ax.add_patch(rect)
    # Generate the color bar
    granularity = 20
    colormap_array = np.linspace(norm.vmin, norm.vmax, granularity) if center_point is None else np.linspace(center_point - max_diff, center_point + max_diff, granularity)
    
    for i, value in enumerate(colormap_array):
        color = cmap(norm(value))
        color = 'silver' if value == 0 else color
        length = 9
        x_offset = 3.5
        y_offset = 7.8
        x_loc = i / granularity * length + x_offset
        width = length / granularity
        height = 0.35
        rect = patches.Rectangle((x_loc, y_offset), width, height,
                                 linewidth=1.5,
                                 edgecolor='gray',
                                 facecolor=color,
                                 alpha=1)

        if i in [0, granularity//4, granularity//2, 3*granularity//4, granularity-1]:
            plt.text(x_loc + width / 2, y_offset - 0.4, f'{value:.1f}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontweight='semibold',
                     fontsize=20, color='k')

        ax.add_patch(rect)

    if property_name is None:
        property_name = property
    plt.text(x_offset + length / 2, y_offset + 1.0,
             property_name,
             horizontalalignment='center',
             verticalalignment='center',
             fontweight='semibold',
             fontsize=20, color='k')
    ax.set_ylim(-0.15, n_row + .1)
    ax.set_xlim(0.85, n_column + 1.1)

    ax.axis('off')
    plt.draw()
    plt.pause(0.001)
    plt.close()
    return fig, ax

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.path import Path
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import os

def periodic_table_dual_plot(plot_df, 
                        property1="Eseg_min1",
                        property2="Eseg_min2",  # New property
                        count_min1=None,
                        count_max1=None,
                        count_min2=None,
                        count_max2=None,
                        center_cm_zero1=False,
                        center_cm_zero2=False,
                        center_point1=None,  # New parameter for arbitrary centering
                        center_point2=None,
                        property_name1=None,
                        property_name2=None,
                        cmap1=plt.cm.Blues,  # Colormap for the first property
                        cmap2=plt.cm.Reds,  # Colormap for the second property
                        element_font_color="darkgoldenrod"):
    module_path = os.path.dirname(os.path.abspath(__file__))
    ptable = pd.read_csv(os.path.join(module_path, 'periodic_table.csv'))
    ptable.index = ptable['symbol'].values
    elem_tracker = ptable['count']
    ptable = ptable[ptable['Z'] <= 92]  # Cap at element 92

    n_row = ptable['row'].max()
    n_column = ptable['column'].max()

    fig, ax = plt.subplots(figsize=(n_column, n_row))
    rows = ptable['row']
    columns = ptable['column']
    symbols = ptable['symbol']
    rw = 0.9  # rectangle width
    rh = rw    # rectangle height

    if count_min1 is None:
        count_min1 = plot_df[property1].min()
    if count_max1 is None:
        count_max1 = plot_df[property1].max()

    # Adjust normalization based on centering preference
    if center_cm_zero1:
        cm_threshold1 = max(abs(count_min1), abs(count_max1))
        norm1 = Normalize(-cm_threshold1, cm_threshold1)
    elif center_point1 is not None:
        # Adjust normalization to center around the arbitrary point
        max_diff = max(center_point1 - count_min1, count_max1 - center_point1)
        norm1 = Normalize(center_point1 - max_diff, center_point1 + max_diff)
    else:
        norm1 = Normalize(vmin=count_min1, vmax=count_max1)

    if count_min2 is None:
        count_min2 = plot_df[property2].min()
    if count_max2 is None:
        count_max2 = plot_df[property2].max()

    # Adjust normalization based on centering preference for the second property
    if center_cm_zero2:
        cm_threshold2 = max(abs(count_min2), abs(count_max2))
        norm2 = Normalize(-cm_threshold2, cm_threshold2)
    elif center_point2 is not None:
        # Adjust normalization to center around the arbitrary point for the second property
        max_diff2 = max(center_point2 - count_min2, count_max2 - center_point2)
        norm2 = Normalize(center_point2 - max_diff2, center_point2 + max_diff2)
    else:
        norm2 = Normalize(vmin=count_min2, vmax=count_max2)

    for row, column, symbol in zip(rows, columns, symbols):
        row = ptable['row'].max() - row
        # Initial color set to 'none' for both properties
        color1, color2 = 'none', 'none'

        if symbol in plot_df.element.unique():
            element_data = plot_df[plot_df["element"] == symbol]
            if property1 in element_data and not element_data[property1].isna().all():
                value1 = element_data[property1].values[0]
                color1 = cmap1(norm1(value1))
            if property2 in element_data and not element_data[property2].isna().all():
                value2 = element_data[property2].values[0]
                color2 = cmap2(norm2(value2))

        # Draw upper right triangle for property1
        triangle1 = patches.Polygon([(column, row), (column + rw, row), (column + rw, row + rh)], 
                                    closed=True, color=color1)
        ax.add_patch(triangle1)
        
        # Draw lower left triangle for property2
        triangle2 = patches.Polygon([(column, row), (column, row + rh), (column + rw, row + rh)], 
                                    closed=True, color=color2)
        ax.add_patch(triangle2)

        # Element symbol
        plt.text(column + rw / 2, row + rh / 2, symbol,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=22,  # Adjusted for visibility
                 fontweight='semibold',
                 color=element_font_color)

    # Generate the color bar
    granularity = 20
    colormap_array = np.linspace(norm1.vmin, norm1.vmax, granularity) if center_point1 is None else np.linspace(center_point1- max_diff, center_point1 + max_diff, granularity)
    
    for i, value in enumerate(colormap_array):
        color = cmap1(norm1(value))
        color = 'silver' if value == 0 else color
        length = 9
        x_offset = 3.5
        y_offset = 7.8
        x_loc = i / granularity * length + x_offset
        width = length / granularity
        height = 0.35
        rect = patches.Rectangle((x_loc, y_offset), width, height,
                                 linewidth=1.5,
                                 edgecolor='gray',
                                 facecolor=color,
                                 alpha=1)

        if i in [0, granularity//4, granularity//2, 3*granularity//4, granularity-1]:
            plt.text(x_loc + width / 2, y_offset - 0.4, f'{value:.1f}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontweight='semibold',
                     fontsize=20, color='k')

        ax.add_patch(rect)

    if property_name1 is None:
        property_name1 = property_name1
    plt.text(x_offset + length / 2, y_offset + 1.0,
             property_name1,
             horizontalalignment='center',
             verticalalignment='center',
             fontweight='semibold',
             fontsize=20, color='k')

    ax.set_ylim(-0.15, n_row + .1)
    ax.set_xlim(0.85, n_column + 1.1)
    ax.axis('off')
    
    plt.draw()
    plt.pause(0.001)
    plt.close()
    return fig, ax
