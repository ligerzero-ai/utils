import os

from pymatgen.core import Structure, Element
from pymatgen.command_line.chargemol_caller import (
    ChargemolAnalysis as PMGChargemolAnalysis,
)

import pandas as pd
import numpy as np

import utils.generic as gen_tools

from utils.parallel import parallelise
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import time


def get_stats(property_list, property_str):
    """
    Calculate statistical properties of a list of values.
    Parameters:
    property_list (list): A list of numerical values for which statistics are calculated.
    property_str (str): A string prefix to be used in the resulting statistical property names.
    Returns:
    dict: A dictionary containing statistical properties with keys in the format:
          "{property_str}_{statistic}" where statistic can be "std" (standard deviation),
          "mean" (mean), "min" (minimum), and "max" (maximum).
    Example:
    >>> values = [1, 2, 3, 4, 5]
    >>> get_stats(values, "example")
    {'example_std': 1.4142135623730951,
     'example_mean': 3.0,
     'example_min': 1,
     'example_max': 5}
    """
    return {
        f"{property_str}_std": np.std(property_list),
        f"{property_str}_mean": np.mean(property_list),
        f"{property_str}_min": np.min(property_list),
        f"{property_str}_max": np.max(property_list),
    }


def check_chargemol_output_present(
    directory,
    required_files=[
        "DDEC6_even_tempered_atomic_spin_moments.xyz",
        "DDEC6_even_tempered_net_atomic_charges.xyz",
        "DDEC_atomic_Rfourth_moments.xyz",
        "overlap_populations.xyz",
        "DDEC6_even_tempered_bond_orders.xyz",
        "DDEC_atomic_Rcubed_moments.xyz",
        "DDEC_atomic_Rsquared_moments.xyz",
        "POTCAR",
    ],
):
    missing_files = [
        file
        for file in required_files
        if not os.path.exists(os.path.join(directory, file))
    ]
    if missing_files:
        return False
    else:
        return True  # All required files are present
    
def get_cleavage_planes(struct, axis, tolerance):
    """
    Get the cleavage planes for a given structure and axis.
    Parameters:
    struct (Structure): A pymatgen Structure object.
    axis (int): The axis to project the structure on.
    tolerance (float): The tolerance for the unique values in the nth value. (Cartesian length)
    Returns:
    list: A list of cleavage planes.
    """
    atomic_layers = get_unique_values_in_nth_value(
        struct.cart_coords, axis, tolerance=tolerance
    )
    cp_list = compute_average_pairs(atomic_layers)
    return cp_list

def summarise_DDEC_data(directory, bond_order_threshold=0.05):
    if not check_chargemol_output_present(directory):
        # Some files are missing, return a DataFrame with NaN values and the filepath
        columns = [
            "bond_order_std",
            "bond_order_mean",
            "bond_order_min",
            "bond_order_max",
            "n_bonds",
            "element",
            "bond_order_sums",
            "ddec_charges",
            "cm5_charges",
            "ddec_rcubed_moments",
            "ddec_rfourth_moments",
            "ddec_spin_moments",
            "dipoles",
            "charge_transfer",
            "partial_charge",
        ]
        empty_data = [[np.nan] * len(columns)]
        ddec_df = pd.DataFrame(empty_data, columns=columns)
        ddec_df["filepath"] = directory
    else:
        ca = PMGChargemolAnalysis(directory, run_chargemol=False)
        bo_df = []
        element_list = []

        for entries in ca.bond_order_dict:
            df = pd.DataFrame(ca.bond_order_dict[entries]["bonded_to"])
            df_thres = df[df["bond_order"] > bond_order_threshold]
            # This is a failsafe because certain atoms just don't bond (e.g. He/Ar)
            if len(df_thres) == 0:
                df_thres = df
                bo_stats_df = get_stats(df_thres.bond_order.tolist(), "bond_order")
                bo_stats_df = pd.DataFrame.from_dict(
                    bo_stats_df, orient="index", columns=[str(entries)]
                ).T
                bo_stats_df["n_bonds"] = 0
            else:
                bo_stats_df = get_stats(df_thres.bond_order.tolist(), "bond_order")
                bo_stats_df = pd.DataFrame.from_dict(
                    bo_stats_df, orient="index", columns=[str(entries)]
                ).T
                bo_stats_df["n_bonds"] = len(df_thres)
            bo_df.append(bo_stats_df)
            element_symbol = ca.bond_order_dict[entries]["element"].symbol
            element_list.append(element_symbol)

        ddec_df = pd.concat(bo_df)
        ddec_df["filepath"] = directory
        ddec_df["element"] = element_list
        ddec_df["bond_order_sums"] = ca.bond_order_sums
        ddec_df["ddec_charges"] = ca.ddec_charges
        try:
            ddec_df["cm5_charges"] = ca.cm5_charges
        except Exception as e:
            print(f"{directory}: FAILED DUE TO EXCEPTION {e}")
            ddec_df["cm5_charges"] = np.nan
        ddec_df["ddec_rcubed_moments"] = ca.ddec_rcubed_moments
        ddec_df["ddec_rfourth_moments"] = ca.ddec_rfourth_moments
        ddec_df["ddec_spin_moments"] = ca.ddec_spin_moments
        ddec_df["dipoles"] = ca.dipoles
        ddec_df["charge_transfer"] = [
            ca.get_charge_transfer(i) for i in ca.bond_order_dict
        ]
        ddec_df["partial_charge"] = [
            ca.get_partial_charge(i) for i in ca.bond_order_dict
        ]

    return ddec_df


def get_solute_summary_DDEC_data(
    directory, bond_order_threshold=0.05, base_solute="Fe"
):
    df = summarise_DDEC_data(
        directory=directory, bond_order_threshold=bond_order_threshold
    )
    df = df[df["element"] == base_solute]
    return df


class DatabaseGenerator:

    def __init__(self, parent_dir):
        self.parent_dir = parent_dir

    def build_database(
        self,
        target_directory=None,
        extract_directories=False,
        cleanup=False,
        keep_filenames_after_cleanup=[],
        keep_filename_patterns_after_cleanup=[],
        max_dir_count=None,
        df_filename=None,
    ):

        start_time = time.time()

        if target_directory:
            dirs = find_chargemol_directories(
                parent_dir=target_directory, extract_tarballs=extract_directories, all_present=False
            )
        else:
            dirs = find_chargemol_directories(
                parent_dir=self.parent_dir, extract_tarballs=extract_directories, all_present=False
            )

        print(
            f"The total number of vasp directories that we are building the database out of is {len(dirs)}"
        )

        if max_dir_count:
            pkl_filenames = []
            for i, chunks in enumerate(gen_tools.chunk_list(dirs, max_dir_count)):
                step_time = time.time()
                df = pd.concat(parallelise(summarise_DDEC_data, chunks))
                if df_filename:
                    db_filename = f"{i}_{df_filename}.pkl"
                else:
                    db_filename = f"{i}.pkl"
                pkl_filenames.append(os.path.join(self.parent_dir, db_filename))
                df.to_pickle(os.path.join(self.parent_dir, db_filename))
                step_taken_time = np.round(step_time - time.time(), 3)
                print(
                    f"Step {i}: {step_taken_time} seconds taken for {len(chunks)} parse steps"
                )

            df = pd.concat([pd.read_pickle(partial_df) for partial_df in pkl_filenames])

        else:
            df = pd.concat(parallelise(summarise_DDEC_data, dirs))
        if df_filename:
            df.to_pickle(os.path.join(self.parent_dir, df_filename))
        else:
            df.to_pickle(os.path.join(self.parent_dir, f"vasp_database.pkl"))
        end_time = time.time()
        elapsed_time = end_time - start_time

        # not optional - keep the tarballs/zips..
        keep_filename_patterns_after_cleanup += ".tar.gz"
        keep_filename_patterns_after_cleanup += ".tar.bz2"
        keep_filename_patterns_after_cleanup += ".zip"

        if cleanup:
            gen_tools.cleanup_dir(
                directory_path=dirs, keep=True, files=[], file_patterns=[]
            )
            parallelise(
                gen_tools.cleanup_dir,
                dirs,
                [True] * len(dirs),
                keep_filenames_after_cleanup * len(dirs),
                keep_filename_patterns_after_cleanup * len(dirs),
            )

        print("Elapsed time:", np.round(elapsed_time, 3), "seconds")

        return df


class ChargemolAnalysis:
    def __init__(self, directory, extract_dir=False):
        self.directory = directory
        self._struct = None
        self._bond_matrix = None
        if extract_dir:
            directory = find_chargemol_directories(directory)[0]
        if check_valid_chargemol_output(
            os.path.join(directory, "VASP_DDEC_analysis.output")
        ):
            self.parse_DDEC6_analysis_output()
        else:
            print(
                "No valid output available! Try extracting any tarballs? Set extract_dir=True"
            )

    def parse_DDEC6_analysis_output(self):
        struct, bond_matrix = parse_DDEC6_analysis_output(
            os.path.join(self.directory, "VASP_DDEC_analysis.output")
        )
        self.struct = struct
        self.bond_matrix = bond_matrix
        return struct, bond_matrix

    # Getter for struct attribute
    def get_struct(self):
        return self._struct

    # Setter for struct attribute
    def set_struct(self, struct):
        self._struct = struct

    # Getter for bond_matrix attribute
    def get_bond_matrix(self):
        return self._bond_matrix

    # Setter for bond_matrix attribute
    def set_bond_matrix(self, bond_matrix):
        self._bond_matrix = bond_matrix

    def plot_ANSBO_profile(self):
        plot_ANSBO_profile_and_structure(self.struct, self.bond_matrix)

    def get_ANSBO_profile(self, axis=2, tolerance=0.1):
        return get_ANSBO_all_cleavage_planes(
            self.struct, self.bond_matrix, axis=axis, tolerance=tolerance
        )
    def get_cleavage_planes(self, axis=2, tolerance=0.1):
        return get_cleavage_planes(self.struct, axis, tolerance)
    
    def get_min_ANSBO(self, axis=2, tolerance=0.1):
        return min(
            get_ANSBO_all_cleavage_planes(
                self.struct, self.bond_matrix, axis=axis, tolerance=tolerance
            )
        )

    def analyse_ANSBO(self, axis=2, tolerance=0.1):
        return analyse_ANSBO(self.directory, axis=axis, tolerance=tolerance)


def analyse_ANSBO(directory, axis=2, tolerance=0.1):
    """ """
    struct, bond_matrix = parse_DDEC6_analysis_output(
        os.path.join(directory, "VASP_DDEC_analysis.output")
    )
    atomic_layers = get_unique_values_in_nth_value(
        struct.cart_coords, axis, tolerance=tolerance
    )
    cp_list = get_cleavage_planes(struct, axis, tolerance)
    ANSBO_profile = get_ANSBO_all_cleavage_planes(
        struct, bond_matrix, axis=axis, tolerance=tolerance
    )

    results_dict = {
        "layer_boundaries": atomic_layers,
        "cleavage_coord": cp_list,
        "ANSBO_profile": ANSBO_profile,
    }
    return results_dict


def find_chargemol_directories(
    parent_dir,
    filenames=[
        "DDEC6_even_tempered_atomic_spin_moments.xyz",
        "DDEC6_even_tempered_net_atomic_charges.xyz",
        "DDEC_atomic_Rfourth_moments.xyz",
        "overlap_populations.xyz",
        "DDEC6_even_tempered_bond_orders.xyz",
        "DDEC_atomic_Rcubed_moments.xyz",
        "DDEC_atomic_Rsquared_moments.xyz",
        "POTCAR",
    ],
    all_present=False,
    extract_tarballs=False,
    only_valid_output=True,
):
    if extract_tarballs:
        gen_tools.find_and_extract_files_from_tarballs_parallel(
            parent_dir=parent_dir,
            extension=".tar.gz",
            filenames=filenames,
            suffix=None,
            prefix=None,
        )
    
    directories = gen_tools.find_directories_with_files(
        parent_dir=parent_dir, filenames=filenames, all_present=all_present
    )
    
    if only_valid_output:
        converged_list = []
        non_converged_list = []
        for dir in directories:
            file = os.path.join(dir, "VASP_DDEC_analysis.output")
            print(file)
            if check_valid_chargemol_output(file):
                converged_list.append(dir)
            else:
                print(f"check output for {dir} failed!")
                non_converged_list.append(dir)
        directories = converged_list
    return directories


def parse_DDEC6_analysis_output(filename):
    """
    Parses VASP_DDEC_analysis.output files and returns a Structure object and bond matrix.

    Args:
        filepaths (str or list): The path(s) to the DDEC6 output file(s) to be parsed.

    Returns:
        tuple: A tuple containing the Structure object and bond matrix.
            - The Structure object represents the atomic structure of the system
            and contains information about the lattice, atomic coordinates,
            and atomic numbers.
            - The bond matrix is a DataFrame that provides information about the
            bonding interactions in the system, including bond indices, bond lengths,
            and other properties.

    Raises:
        FileNotFoundError: If the specified file(s) do not exist.

    Example:
        filepaths = ["output1.txt", "output2.txt"]
        structure, bond_matrix = parse_DDEC6(filepaths)
        print(structure)
        print(bond_matrix)

    Note:
        - The function reads the specified DDEC6 output file(s) and extracts relevant
        information to create a Structure object and bond matrix.
        - The function expects the DDEC6 output files to be in a specific format and
        relies on certain trigger lines to identify the relevant sections.
        - The structure lattice is parsed from the lines between the "vectors" and
        "direct_coords" triggers.
        - The atomic fractional coordinates are parsed from the lines between the
        "direct_coords" and "totnumA" triggers.
        - The atomic numbers are parsed from the lines between the "(Missing core
        electrons will be inserted using stored core electron reference densities.)"
        and "Finished the check for missing core electrons." triggers.
        - The atomic numbers are converted to element symbols using the pymatgen
        Element.from_Z() method.
        - The Structure object is created using the parsed lattice, atomic numbers,
        and fractional coordinates.
        - The bond matrix is parsed from the lines between the "The final bond pair
        matrix is" and "The legend for the bond pair matrix follows:" triggers.
        - The bond matrix is returned as a pandas DataFrame with the specified column
        names.

    """
    flist = open(filename).readlines()

    bohr_to_angstrom_conversion_factor = 0.529177
    structure_lattice = gen_tools.parse_lines(
        flist, trigger_start="vectors", trigger_end="direct_coords"
    )[0]
    structure_lattice = np.array(
        [list(map(float, line.split())) for line in structure_lattice]
    )
    structure_lattice = structure_lattice * bohr_to_angstrom_conversion_factor

    structure_frac_coords = gen_tools.parse_lines(
        flist, trigger_start="direct_coords", trigger_end="totnumA"
    )[0]
    structure_frac_coords = [
        np.array([float(coord) for coord in entry.split()])
        for entry in structure_frac_coords
    ]

    # Convert atomic numbers to element symbols
    structure_atomic_no = gen_tools.parse_lines(
        flist,
        trigger_start="(Missing core electrons will be inserted using stored core electron reference densities.)",
        trigger_end=" Finished the check for missing core electrons.",
    )
    structure_atomic_no = [
        Element.from_Z(int(atomic_number.split()[1])).symbol
        for atomic_number in structure_atomic_no[0]
    ]

    structure = Structure(structure_lattice, structure_atomic_no, structure_frac_coords)

    data_column_names = [
        "atom1",
        "atom2",
        "repeata",
        "repeatb",
        "repeatc",
        "min-na",
        "max-na",
        "min-nb",
        "max-nb",
        "min-nc",
        "max-nc",
        "contact-exchange",
        "avg-spin-pol-bonding-term",
        "overlap-population",
        "isoaepfcbo",
        "coord-term-tanh",
        "pairwise-term",
        "exp-term-comb-coord-pairwise",
        "bond-idx-before-self-exch",
        "final_bond_order",
    ]

    bond_matrix = gen_tools.parse_lines(
        flist,
        trigger_start="The final bond pair matrix is",
        trigger_end="The legend for the bond pair matrix follows:",
    )[0]
    bond_matrix = np.array([list(map(float, line.split())) for line in bond_matrix])
    bond_matrix = pd.DataFrame(bond_matrix, columns=data_column_names)

    return structure, bond_matrix


def check_valid_chargemol_output(vasp_ddec_analysis_output_filepath):
    """
    Checks if a VASP DDEC analysis output file indicates successful completion of Chargemol.

    Args:
        vasp_ddec_analysis_output_filepath (str): The path to the VASP DDEC analysis output file.

    Returns:
        bool: True if Chargemol analysis has successfully finished, False otherwise.

    Example:
        output_filepath = "vasp_ddec_analysis_output.txt"
        result = check_valid_chargemol_output(output_filepath)
        if result:
            print("Chargemol analysis finished successfully.")
        else:
            print("Chargemol analysis did not finish.")

    Notes:
        - The function reads the VASP DDEC analysis output file and searches for a specific line
          indicating the completion of the Chargemol analysis.
        - If the specified line is found, the function returns True, indicating successful completion.
        - If the specified line is not found, the function returns False, indicating that the Chargemol
          analysis did not finish or encountered an error.
        - The function assumes that the VASP DDEC analysis output file follows a specific format and
          contains the necessary information.

    """
    convergence = gen_tools.search_line_in_file(
        vasp_ddec_analysis_output_filepath, "Finished chargemol in"
    )

    return convergence


def plot_structure_projection(
    structure,
    projection_axis=[1, 2],
    bond_matrix=None,
    atom_size=250,
    figsize=(8, 6),
    cell_border_colour="r",
    atom_colour_dict=None,
    fontsize=16,
):
    """
    Plots the projection of a pymatgen structure on a 2D plane based on the specified projection axis.

    Parameters:
        structure (pymatgen.core.structure.Structure): The pymatgen Structure object.
        projection_axis (list): A list of two integers specifying the axes for the x and y coordinates.

    Returns:
        None (displays the plot).
    """
    # Extract the atomic coordinates based on the projection axis
    x_coords = [site.coords[projection_axis[0]] for site in structure]
    y_coords = [site.coords[projection_axis[1]] for site in structure]

    # Create the plot
    # plt.figure(figsize=figsize)
    for site in structure:
        species = site.species_string
        if atom_colour_dict is not None:
            color = atom_colour_dict.get(
                species, "b"
            )  # Default to blue if species not in atom_colour_dict
        else:
            color = "b"
        plt.scatter(
            site.coords[projection_axis[0]],
            site.coords[projection_axis[1]],
            color=color,
            s=atom_size,
            edgecolors="black",
        )

    # Set plot title and labels
    plt.title("Projection of the Cell", fontsize=16)
    plt.xlabel(f"Axis {projection_axis[0]} Coordinate", fontsize=12)
    plt.ylabel(f"Axis {projection_axis[1]} Coordinate", fontsize=12)

    # Set plot limits based on the atomic coordinates
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    plt.xlim(x_min - 1, x_max + 1)
    plt.ylim(y_min - 1, y_max + 1)

    if bond_matrix is not None:
        relevant_plot_bonds = bond_matrix[
            (bond_matrix["repeata"] == 0) & (bond_matrix["repeatb"] == 0)
        ]
        for idx, bonds in relevant_plot_bonds.iterrows():
            atom1 = int(bonds["atom1"]) - 1
            atom2 = int(bonds["atom2"]) - 1
            bondstrength = np.round(bonds["final_bond_order"], 2)
            if bondstrength < 0.28:
                c = "r"
            else:
                c = "k"
            c = "k"
            plt.plot(
                [
                    structure[atom1].coords[projection_axis[0]],
                    structure[atom2].coords[projection_axis[0]],
                ],
                [
                    structure[atom1].coords[projection_axis[1]],
                    structure[atom2].coords[projection_axis[1]],
                ],
                "-",
                color=c,
                linewidth=bondstrength / 0.56 * 5,
            )

    # Draw the cell with a black border based on the projection_axis
    lattice_vectors = structure.lattice.matrix[projection_axis]

    # Draw the cell with a border based on the projection_axis
    rect = plt.Rectangle(
        (0, 0),
        structure.lattice.abc[projection_axis[0]],
        structure.lattice.abc[projection_axis[1]],
        edgecolor=cell_border_colour,
        linewidth=3,
        fill=False,
        linestyle="--",
    )
    plt.gca().add_patch(rect)
    plt.gca().set_aspect("equal")
    plt.grid()


def get_unique_values_in_nth_value(arr_list, n, tolerance):
    unique_values = []
    for sublist in arr_list:
        value = sublist[n]
        is_unique = True
        for unique_value in unique_values:
            if np.allclose(value, unique_value, atol=tolerance):
                is_unique = False
                break
        if is_unique:
            unique_values.append(value)
    return np.sort(unique_values)


def compute_average_pairs(lst):
    averages = []
    for i in range(len(lst) - 1):
        average = (lst[i] + lst[i + 1]) / 2
        averages.append(average)
    return averages


def get_ANSBO(structure, bond_matrix, cleavage_plane, axis=2):
    bond_matrix["atom1pos"] = [
        structure[int(x) - 1].coords[axis] for x in bond_matrix["atom1"].values
    ]
    bond_matrix["atom2pos"] = [
        structure[int(x) - 1].coords[axis] for x in bond_matrix["atom2"].values
    ]
    clp_df = bond_matrix[
        (bond_matrix[["atom1pos", "atom2pos"]].max(axis=1) > cleavage_plane)
        & (bond_matrix[["atom1pos", "atom2pos"]].min(axis=1) < cleavage_plane)
    ]
    if axis == 0:
        repeat1 = "repeatb"
        repeat2 = "repeatc"
    elif axis == 1:
        repeat1 = "repeata"
        repeat2 = "repeatc"
    elif axis == 2:
        repeat1 = "repeata"
        repeat2 = "repeatb"

    clp_df = clp_df.copy()[(clp_df[repeat1] == 0) | (clp_df[repeat2] == 0)]
    # We only want to calculate for atoms that exist in cell. This is important for bond order/area normalisation
    clp_df_countonce = clp_df.copy()[(clp_df[repeat1] == 0) & (clp_df[repeat2] == 0)]
    clp_df_counthalf = clp_df.copy()[(clp_df[repeat1] != 0) | (clp_df[repeat2] != 0)]
    # Basic summed bond order over CP
    final_bond_order = (
        clp_df_countonce.final_bond_order.sum()
        + 0.5 * clp_df_counthalf.final_bond_order.sum()
    )
    # N largest
    # final_bond_order = clp_df.nlargest(15, ['final_bond_order'])["final_bond_order"].sum()
    # IMPORTANT: This assumes that the cross sectional area can be calculated this way
    a_fbo = final_bond_order / (
        float(structure.lattice.volume) / float(structure.lattice.abc[axis])
    )
    # print("area of this is %s" % (float(structure.lattice.volume)/float(structure.lattice.c)))
    return a_fbo


def get_ANSBO_all_cleavage_planes(structure, bond_matrix, axis=2, tolerance=0.1):
    cp_list = get_cleavage_planes(structure, axis, tolerance)
    ANSBO_profile = []
    for cp in cp_list:
        ANSBO_profile.append(get_ANSBO(structure, bond_matrix, cp))
    return ANSBO_profile


def plot_ANSBO_profile(structure, bond_matrix, projection_axis=[1, 2]):
    ANSBO_values = get_ANSBO_all_cleavage_planes(
        structure, bond_matrix, projection_axis[-1]
    )
    atomic_layer_coords = get_unique_values_in_nth_value(
        structure.cart_coords, projection_axis[-1], tolerance=0.1
    )
    if len(atomic_layer_coords) != len(ANSBO_values) + 1:
        print("Error: Lengths of the lists are not compatible.")
        return

    # plt.figure(figsize=(3,10))

    # Create lists for the x and y coordinates of the lines
    x_lines = []
    y_lines = []

    # Iterate over the elements of ANSBO_profile
    for i, value in enumerate(ANSBO_values):
        # Append x-coordinates for the horizontal lines
        x_lines.extend([value, value])
        # Append y-coordinates for the horizontal lines
        y_lines.extend([atomic_layer_coords[i], atomic_layer_coords[i + 1]])
        # Append x-coordinates for the vertical lines
        x_lines.append(value)
        # Append y-coordinates for the vertical lines
        y_lines.append(atomic_layer_coords[i + 1])

    # Plotting the lines
    plt.plot(x_lines, y_lines)
    plt.grid()
    # Labeling the axes
    plt.xlabel("ANSBO Profile")
    plt.ylabel("Coordinates (Angstrom)")


def plot_ANSBO_profile_and_structure(
    structure, bond_matrix, write=False, filename="ANSBO.jpg", fontsize=16, atom_colour_dict=None
):
    """
    Plot the structure bond projection and the ANSBO profile side by side.

    Parameters:
        structure (list): The structure data to be plotted.
        bond_matrix (array-like): The bond matrix data for the structure.
        write (bool, optional): If True, the plot will be saved to a file. Default is False.
        filename (str, optional): The filename to save the plot. Default is "ANSBO.jpg".

    Returns:
        None
    """

    # Create a new figure with two subplots side by side
    fig, axs = plt.subplots(
        1, 2, figsize=(10, 20), gridspec_kw={"width_ratios": [2, 1]}
    )

    # Activate the first subplot and call plot_structure_projection
    plt.sca(axs[0])
    plot_structure_projection(
        structure,
        bond_matrix=bond_matrix,
        figsize=(8, 6),
        atom_colour_dict=atom_colour_dict,
    )
    plt.grid(True, which="major", linestyle="-")
    plt.grid(True, which="minor", linestyle="--")
    axs[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axs[0].yaxis.set_minor_locator(ticker.MultipleLocator(1))
    # Activate the second subplot and call plot_ANSBO_profile
    plt.sca(axs[1])
    plot_ANSBO_profile(
        structure, bond_matrix
    )  # Assuming you have defined the plot_ANSBO_profile function
    plt.grid(True, which="major", linestyle="-")
    plt.grid(True, which="minor", linestyle="--")
    axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axs[1].yaxis.set_minor_locator(ticker.MultipleLocator(1))
    # Set the same y-axis limits for both subplots
    axs[1].set_ylim(axs[0].get_ylim())

    # Adjust the spacing between the plots
    plt.subplots_adjust(wspace=0.01)  # Set the desired spacing between the subplots

    # Set titles for the subplots
    axs[0].set_title("Structure Bond Projection", fontsize=fontsize)
    axs[1].set_title("ANSBO Profile", fontsize=fontsize)

    # Optionally, save the plot to a file
    if write:
        plt.savefig(filename)

    # Display the plot
    plt.show()


def plot_ANSBO_profile_and_structure_from_dir(directory, extract_from_tarball=True):
    structure, bond_matrix = parse_DDEC6_analysis_output(
        os.path.join(directory, "VASP_DDEC_analysis.output")
    )
    plot_ANSBO_profile_and_structure(structure, bond_matrix)
    
def get_min_max_coords(structure, axis=2, use_fractional=False):
    """
    Return the minimum and maximum coordinates of all sites along a specified axis.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        The structure to analyze.
    axis : int, default=2
        Coordinate axis: 0 → x/a, 1 → y/b, 2 → z/c.
    use_fractional : bool, default=False
        If True, compute extrema on fractional coordinates; otherwise use Cartesian.

    Returns
    -------
    (min_coord, max_coord) : tuple of floats
        The minimum and maximum coordinate values along the chosen axis.
    """
    # Choose coordinate array
    if use_fractional:
        coords = structure.frac_coords
    else:
        coords = structure.cart_coords

    # Extract the axis values
    axis_vals = coords[:, axis]

    # Compute extrema
    min_val = float(axis_vals.min())
    max_val = float(axis_vals.max())
    return min_val, max_val

def find_extreme_cleavages(
    structure,
    bond_matrix,
    cleavage_planes,
    base_ele="Fe",
    axis=2,
    distance_threshold=4.0,
    away_from_surface_tolerance=3.0,
    use_fractional=False
):
    """
    Loops over cleavage_planes but only considers those within ±distance_threshold
    of any non-base_ele site along given axis and within the cell boundaries
    offset by boundary_tolerance. Computes the distribution df and extracts mean
    and max final_bond_order for each valid plane.

    Parameters
    ----------
    structure : Structure
        Pymatgen structure.
    bond_matrix : DataFrame
        Bond data matrix.
    cleavage_planes : iterable of float
        Candidate plane positions along the axis.
    base_ele : str, default="Fe"
        Element symbol to exclude when testing proximity.
    axis : int, default=2
        Axis index (0=a,1=b,2=c).
    distance_threshold : float, default=4.0
        Max distance from non-base_ele site to plane to consider.
    boundary_tolerance : float, default=0.0
        Distance from global min/max along axis to exclude near-boundary planes.
    use_fractional : bool, default=False
        Whether to compute global boundaries in fractional coords.

    Returns
    -------
    (min_max_bo, plane_min_max), (min_mean_bo, plane_min_mean), df_at_plane_min_max
    """
    # Compute global boundaries with optional fractional coords
    min_bound, max_bound = get_min_max_coords(
        structure, axis=axis, use_fractional=use_fractional
    )
    # Adjust by boundary_tolerance
    low_limit = min_bound + away_from_surface_tolerance
    high_limit = max_bound - away_from_surface_tolerance
    
    # Precompute non-base_ele positions along axis (Cartesian)
    non_idx = [i for i, site in enumerate(structure) if site.species_string != base_ele]
    non_coords = [structure[i].coords[axis] for i in non_idx]

    stats = []
    for plane in cleavage_planes:
        # skip if outside boundary limits
        if plane < low_limit or plane > high_limit:
            continue
        # skip if no non-base_ele within threshold
        if not any(abs(z - plane) <= distance_threshold for z in non_coords):
            continue

        df = get_ANSBO_distribution(
            structure=structure,
            bond_matrix=bond_matrix,
            cleavage_plane=plane,
            axis=axis
        )
        mean_bo = df["final_bond_order"].mean()
        max_bo = df["final_bond_order"].max()
        stats.append((plane, mean_bo, max_bo, df))

    if not stats:
        raise ValueError("No cleavage planes passed the filters.")

    # Select extremes
    plane_min_max, _, min_max_bo, df_min_max = min(stats, key=lambda x: x[2])
    plane_min_mean, min_mean_bo, _, _ = min(stats, key=lambda x: x[1])

    return (min_max_bo, plane_min_max), (min_mean_bo, plane_min_mean), df_min_max
    
def get_ANSBO_distribution(structure,
                           bond_matrix,
                           cleavage_plane,
                           axis=2,
                           zero_threshold = 0.05):
    """
    Returns a DataFrame of bonds crossing the cleavage plane, including their bond orders,
    weights (1 for bonds fully in cell, 0.5 for periodic images), contributions,
    and normalized contributions per unit area along the given axis.

    Parameters:
    - structure: A pymatgen Structure object with attributes .coords and .lattice
    - bond_matrix: A DataFrame containing at least the columns ['atom1', 'atom2', 'final_bond_order',
      'repeata', 'repeatb', 'repeatc']
    - cleavage_plane: Coordinate value along `axis` at which the plane lies
    - axis: The Cartesian axis index (0 for a, 1 for b, 2 for c)

    Returns:
    - df: A pandas DataFrame with rows for each bond crossing the plane and columns:
        ['atom1', 'atom2', 'atom1pos', 'atom2pos', 'repeat*', 'final_bond_order',
         'weight', 'contribution', 'normalized_contribution']
    """
    # Copy input DataFrame
    df = bond_matrix.copy()
    
    # Compute atomic positions along the specified axis
    df["atom1pos"] = df["atom1"].astype(int).apply(lambda i: structure[i-1].coords[axis])
    df["atom2pos"] = df["atom2"].astype(int).apply(lambda i: structure[i-1].coords[axis])

    # Select bonds that span the cleavage plane
    mask_cross = (
        df[["atom1pos", "atom2pos"]].max(axis=1) > cleavage_plane
    ) & (
        df[["atom1pos", "atom2pos"]].min(axis=1) < cleavage_plane
    )
    df = df.loc[mask_cross].copy()

    # Identify periodic repeat directions based on axis
    if axis == 0:
        repeat1, repeat2 = "repeatb", "repeatc"
    elif axis == 1:
        repeat1, repeat2 = "repeata", "repeatc"
    else:
        repeat1, repeat2 = "repeata", "repeatb"

    df_filt = df[df["final_bond_order"] > zero_threshold]
    return df_filt

def get_non_element_idx(structure, base_ele="Fe"):
    # Compare each site.species_string to base_ele
    return [
        i
        for i, site in enumerate(structure)
        if site.species_string != base_ele
    ]