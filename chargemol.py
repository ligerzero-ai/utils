
import os

from pymatgen.core import Structure, Element

import pandas as pd
import numpy as np

from utils.generic import parse_lines, search_line_in_file, find_directories_with_files

import matplotlib.pyplot as plt

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
  structure_lattice = parse_lines(flist, trigger_start="vectors", trigger_end="direct_coords")[0]
  structure_lattice = np.array([list(map(float, line.split())) for line in structure_lattice])
  structure_lattice = structure_lattice * bohr_to_angstrom_conversion_factor
  
  structure_frac_coords = parse_lines(flist, trigger_start="direct_coords", trigger_end="totnumA")[0]
  structure_frac_coords = [np.array([float(coord) for coord in entry.split()]) for entry in structure_frac_coords]
  
  # Convert atomic numbers to element symbols
  structure_atomic_no = parse_lines(flist, trigger_start="(Missing core electrons will be inserted using stored core electron reference densities.)", trigger_end=" Finished the check for missing core electrons.")
  structure_atomic_no = [Element.from_Z(int(atomic_number.split()[1])).symbol for atomic_number in structure_atomic_no[0]]
  
  structure = Structure(structure_lattice, structure_atomic_no, structure_frac_coords)

  data_column_names = ['atom1',\
              'atom2',\
              'repeata',\
              'repeatb',\
              'repeatc',\
              'min-na',\
              'max-na',\
              'min-nb',\
              'max-nb',\
              'min-nc',\
              'max-nc',\
              'contact-exchange',\
              'avg-spin-pol-bonding-term',\
              'overlap-population',\
              'isoaepfcbo',\
              'coord-term-tanh',\
              'pairwise-term',\
              'exp-term-comb-coord-pairwise',\
              'bond-idx-before-self-exch',\
              'final_bond_order']

  bond_matrix = parse_lines(flist, trigger_start="The final bond pair matrix is", trigger_end="The legend for the bond pair matrix follows:")[0]
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
    convergence = search_line_in_file(vasp_ddec_analysis_output_filepath, "Finished chargemol in")

    return convergence
  
def find_chargemol_dirs(filepath):
    """
    Find directories with Chargemol output files in the specified filepath.

    Args:
        filepath (str): The path to the directory to search for Chargemol output files.

    Returns:
        tuple: A tuple containing two lists:
            - The first list contains paths to directories with Chargemol output files that indicate successful completion.
            - The second list contains paths to directories with Chargemol output files that did not indicate successful completion.

    Example:
        directory_path = "/path/to/directory"
        converged_dirs, non_converged_dirs = find_chargemol_dirs(directory_path)
        print("Converged directories:")
        for converged_dir in converged_dirs:
            print(converged_dir)
        print("Non-converged directories:")
        for non_converged_dir in non_converged_dirs:
            print(non_converged_dir)

    Notes:
        - The function searches for directories in the specified filepath that contain Chargemol output files.
        - The Chargemol output files are expected to have the name "VASP_DDEC_analysis.output" and be located
          in the same directory as an "INCAR" file.
        - The function uses the helper function `find_filepaths_in_dir_with_files` to find directories with "INCAR" files.
        - For each directory found, the function checks if the corresponding "VASP_DDEC_analysis.output" file indicates
          successful completion using the `check_valid_chargemol_output` function.
        - Directories with successful Chargemol completion are added to the converged list, while directories without
          successful completion are added to the non-converged list.
        - The function returns the converged and non-converged lists as a tuple.

    """
    whole_list = find_directories_with_files(filepath, ["INCAR"])
    whole_list = [os.path.join(os.path.dirname(path), "VASP_DDEC_analysis.output") for path in whole_list]
    converged_list = []
    non_converged_list = []
    for file in whole_list:
        if check_valid_chargemol_output(file):
            converged_list.append(file)
        else:
            non_converged_list.append(file)
    
    return converged_list, non_converged_list
    
def run_scrape(filepath):
    """
    Run the Chargemol scraping process on the specified filepath.

    Args:
        filepath (str): The path to the directory to run the Chargemol scraping process on.

    Returns:
        tuple: A tuple containing two elements:
            - The first element is a list of filepaths to directories with successful Chargemol completion.
            - The second element is the result of parsing the Chargemol output files.

    Example:
        directory_path = "/path/to/directory"
        filepaths, results = run_scrape(directory_path)
        print("Filepaths with successful Chargemol completion:")
        for filepath in filepaths:
            print(filepath)
        print("Parsing results:")
        print(results)

    Notes:
        - The function runs the Chargemol scraping process on the specified filepath.
        - It first calls the `find_chargemol_dirs` function to find directories with Chargemol output files,
          separating them into filepaths with successful completion and non-converged filepaths.
        - The successful completion filepaths are then passed to the `parse_DDEC6` function to parse the Chargemol output files.
        - The parsing results are returned along with the filepaths.

    """
    filepaths, non_converged_filepaths = find_chargemol_dirs(filepath)
    results = parse_DDEC6_analysis_output(filepaths)
    # Pad the results with None if the lengths don't match
    max_length = len(filepaths+non_converged_filepaths)
    results += [None] * (max_length - len(results))
    return filepaths, results

def plot_structure_projection(structure,
                              projection_axis = [1, 2], 
                              bond_matrix = None,
                              atom_size=250,
                              figsize=(8, 6),
                              cell_border_colour = "r",
                              atom_colour_dict = {},
                              fontsize=16):
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
        color = atom_colour_dict.get(species, 'b')  # Default to blue if species not in atom_colour_dict
        plt.scatter(site.coords[projection_axis[0]], site.coords[projection_axis[1]], color=color, s=atom_size, edgecolors='black')

    # Set plot title and labels
    plt.title('Projection of the Cell', fontsize=16)
    plt.xlabel(f'Axis {projection_axis[0]} Coordinate', fontsize=12)
    plt.ylabel(f'Axis {projection_axis[1]} Coordinate', fontsize=12)

    # Set plot limits based on the atomic coordinates
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    plt.xlim(x_min - 1, x_max + 1)
    plt.ylim(y_min - 1, y_max + 1)
    
    if bond_matrix is not None:
        relevant_plot_bonds = bond_matrix[(bond_matrix['repeata'] == 0) & (bond_matrix['repeatb'] == 0)]
        for idx, bonds in relevant_plot_bonds.iterrows():        
            atom1 = int(bonds["atom1"])-1
            atom2 = int(bonds["atom2"])-1
            x1 = structure[atom1].coords[0]
            y1 = structure[atom1].coords[1]
            z1 = structure[atom1].coords[2]
            x2 = structure[atom2].coords[0]
            y2 = structure[atom2].coords[1]
            z2 = structure[atom2].coords[2]
            bondstrength = np.round(bonds["final_bond_order"],2)
            if bondstrength < 0.28:
                c = 'r'
            else:
                c = 'k'
            c = "k"
            plt.plot([structure[atom1].coords[projection_axis[0]],structure[atom2].coords[projection_axis[0]]],
                    [structure[atom1].coords[projection_axis[1]],structure[atom2].coords[projection_axis[1]]],
                    '-',
                    color=c,
                    linewidth=bondstrength/0.56*5)
        
    # Draw the cell with a black border based on the projection_axis
    lattice_vectors = structure.lattice.matrix[projection_axis]

    # Draw the cell with a border based on the projection_axis
    rect = plt.Rectangle((0,0),
                         structure.lattice.abc[projection_axis[0]],
                         structure.lattice.abc[projection_axis[1]],
                         edgecolor=cell_border_colour,
                         linewidth=3,
                         fill=False,
                         linestyle = '--')
    plt.gca().add_patch(rect)
    plt.gca().set_aspect('equal')
    plt.grid()