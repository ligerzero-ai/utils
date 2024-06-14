import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm

def plot_structure_projection(structure,
                              projection_axis=[1, 2],
                              bond_matrix=None,
                              atom_size=250,
                              figsize=(8, 6),
                              cell_border_colour="r",
                              no_fill_elements=["Fe"],
                              fill_color="red",
                              atom_size_dict={},
                              fontsize=16,
                              values_list=None,
                              title=None,
                              cmap='viridis',
                              colorbar_label=r"$\rm{E}_{seg}$",
                              xlabel_fontsize=None,
                              ylabel_fontsize=None,
                              title_fontsize=None,
                              colorbar_fontsize=None,
                              colorbar_ticks_fontsize=None,
                              center_colorbar_at_zero=True):
    """
    Plots the projection of a pymatgen structure on a 2D plane based on the specified projection axis.

    Parameters:
        structure (pymatgen.core.structure.Structure): The pymatgen Structure object.
        projection_axis (list): A list of two integers specifying the axes for the x and y coordinates.
        bond_matrix (pd.DataFrame): DataFrame containing bond information.
        atom_size (float): Default size of the atoms in the plot.
        figsize (tuple): Size of the figure.
        cell_border_colour (str): Color of the cell border.
        no_fill_elements (list): List of elements for which to have no fill.
        fill_color (str): Color for elements other than those specified in no_fill_elements.
        atom_size_dict (dict): Dictionary mapping element symbols to custom sizes for atoms.
        fontsize (int): Font size for title and labels.
        values_list (list): List of values to be mapped onto the colormap.
        title (str): Plot title.
        cmap (str): Colormap for atom colors.
        colorbar_label (str): Label for the colorbar.
        xlabel_fontsize (int or None): Font size for xlabel. If None, use the default fontsize.
        ylabel_fontsize (int or None): Font size for ylabel. If None, use the default fontsize.
        title_fontsize (int or None): Font size for title. If None, use the default fontsize.
        colorbar_fontsize (int or None): Font size for colorbar legend. If None, use the default fontsize.
        colorbar_ticks_fontsize (int or None): Font size for colorbar ticks. If None, use the default fontsize.

    Returns:
        None (displays the plot).
    """
    # Extract the atomic coordinates based on the projection axis
    x_coords = [site.coords[projection_axis[0]] for site in structure]
    y_coords = [site.coords[projection_axis[1]] for site in structure]

    # Create the plot
    fig = plt.figure(figsize=figsize)

    if values_list is not None:
        # Adjust vmin and vmax based on the absolute maximum value for symmetry
        max_abs_value = max(abs(min(values_list)), abs(max(values_list)))
        
        if center_colorbar_at_zero:
            norm = TwoSlopeNorm(vmin=-max_abs_value, vcenter=0, vmax=max_abs_value)
        else:
            norm = Normalize(vmin=min(values_list), vmax=max(values_list))
            
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Required for ScalarMappable to work

    for i, site in enumerate(structure):
        species = site.species_string
        if species in no_fill_elements:
            color = 'none'  # No fill for specified elements
        else:
            if values_list is not None:
                color = sm.to_rgba(values_list[i])
            else:
                color = fill_color

        # Use custom size if available, otherwise use the default size
        size = atom_size_dict.get(species, atom_size)
        plt.scatter(site.coords[projection_axis[0]], site.coords[projection_axis[1]], color=color, s=size,
                    edgecolors='black')

    # Set plot title and labels
    if title is not None:
        plt.title(title, fontsize=title_fontsize or fontsize)
    # plt.xlabel(f'Axis {projection_axis[0]} Coordinate', fontsize=xlabel_fontsize or fontsize-4)
    # plt.ylabel(f'Axis {projection_axis[1]} Coordinate', fontsize=ylabel_fontsize or fontsize-4)

    # Set plot limits based on the atomic coordinates
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    plt.xlim(x_min - 1, x_max + 1)
    plt.ylim(y_min - 1, y_max + 1)

    if bond_matrix is not None:
        relevant_plot_bonds = bond_matrix[(bond_matrix['repeata'] == 0) & (bond_matrix['repeatb'] == 0)]
        for idx, bonds in relevant_plot_bonds.iterrows():
            atom1 = int(bonds["atom1"]) - 1
            atom2 = int(bonds["atom2"]) - 1
            bondstrength = np.round(bonds["final_bond_order"], 2)
            if bondstrength < 0.28:
                c = 'r'
            else:
                c = 'k'
            c = "k"
            plt.plot([structure[atom1].coords[projection_axis[0]], structure[atom2].coords[projection_axis[0]]],
                     [structure[atom1].coords[projection_axis[1]], structure[atom2].coords[projection_axis[1]]],
                     '-',
                     color=c,
                     linewidth=bondstrength / 0.56 * 5)

    # Draw the cell with a black border based on the projection_axis
    lattice_vectors = structure.lattice.matrix[projection_axis]

    # Draw the cell with a border based on the projection_ax|is
    rect = plt.Rectangle((0, 0),
                         structure.lattice.abc[projection_axis[0]],
                         structure.lattice.abc[projection_axis[1]],
                         edgecolor=cell_border_colour,
                         linewidth=3,
                         fill=False,
                         linestyle='--')
    plt.gca().add_patch(rect)
    plt.gca().set_aspect('equal')
    plt.grid()

    # Add colorbar
    if colorbar_fontsize is not None and values_list is not None:
        cbar_ax = fig.add_axes([0.55, 0.1, 0.005, 0.8])  # Adjust these values to position the colorbar as needed
        cbar = plt.colorbar(sm, cax=cbar_ax, label=colorbar_label)
        cbar.set_label(colorbar_label, fontsize=colorbar_fontsize)
        if colorbar_ticks_fontsize is not None:
            cbar.ax.tick_params(labelsize=colorbar_ticks_fontsize)
    
    plt.show()
