from pymatgen.core import Structure
import pymatgen.transformations.site_transformations as transform
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Potcar, Incar, Kpoints
from pymatgen.io.vasp.outputs import Outcar
import numpy as np

# RIPPED FROM MPINTERFACES

def center_slab(structure):
    """
    Centers the atoms in a slab structure around 0.5
    fractional height.

    Args:
        structure (Structure): Structure to center
    Returns:
        Centered Structure object.
    """

    center = np.average([s.frac_coords[2] for s in structure.sites])
    translation = (0, 0, 0.5 - center)
    structure.translate_sites(range(len(structure.sites)), translation)
    return structure

def get_rotation_matrix(axis, theta):
    """
    Find the rotation matrix associated with counterclockwise rotation
    about the given axis by theta radians.
    Credit: http://stackoverflow.com/users/190597/unutbu

    Args:
        axis (list): rotation axis of the form [x, y, z]
        theta (float): rotational angle in radians

    Returns:
        array. Rotation matrix.
    """

    axis = np.array(list(axis))
    axis = axis / np.linalg.norm(axis)
    axis *= -np.sin(theta/2.0)
    a = np.cos(theta/2.0)
    b, c, d = tuple(axis.tolist())
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def align_axis(structure, axis='c', direction=(0, 0, 1)):
    """
    Rotates a structure so that the specified axis is along
    the [001] direction. This is useful for adding vacuum, and
    in general for using vasp compiled with no z-axis relaxation.

    Args:
        structure (Structure): Pymatgen Structure object to rotate.
        axis: Axis to be rotated. Can be 'a', 'b', 'c', or a 1x3 vector.
        direction (vector): Final axis to be rotated to.
    Returns:
        structure. Rotated to align axis along direction.
    """

    if axis == 'a':
        axis = structure.lattice._matrix[0]
    elif axis == 'b':
        axis = structure.lattice._matrix[1]
    elif axis == 'c':
        axis = structure.lattice._matrix[2]
    proj_axis = np.cross(axis, direction)
    if not(proj_axis[0] == 0 and proj_axis[1] == 0):
        theta = (
            np.arccos(np.dot(axis, direction)
            / (np.linalg.norm(axis) * np.linalg.norm(direction)))
        )
        R = get_rotation_matrix(proj_axis, theta)
        rotation = SymmOp.from_rotation_and_translation(rotation_matrix=R)
        structure.apply_operation(rotation)
    return structure

def add_vacuum(structure, vacuum):
    """
    Adds padding to a slab or 2D material.

    Args:
        structure (Structure): Structure to add vacuum to
        vacuum (float): Vacuum thickness to add in Angstroms
    Returns:
        Structure object with vacuum added.
    """
    structure = align_axis(structure)
    coords = [s.coords for s in structure.sites]
    species = [s.specie for s in structure.sites]
    lattice = structure.lattice.matrix
    lattice.setflags(write=1)
    lattice_C = lattice
    lattice_C[2][2] += vacuum
    structure = Structure(lattice_C, species, coords, coords_are_cartesian=True)
    return center_slab(structure)

def cleave_sites(structure, cleave_line_coord, vacuum_size):
    site_list = []; site_list2 = []
    for idx, sites in enumerate(structure):
        if sites.frac_coords[-1] > cleave_line_coord:
            #print(idx)
            site_list.append(idx)
        else:
            #print(idx)
            site_list2.append(idx)
    transformation_shift_up = transform.TranslateSitesTransformation(site_list,(0,0,vacuum_size/2),vector_in_frac_coords=False)
    transformation_shift_down = transform.TranslateSitesTransformation(site_list2,(0,0,-vacuum_size/2),vector_in_frac_coords=False)
    cleaved_cell = transformation_shift_up.apply_transformation(structure)
    cleaved_cell = transformation_shift_down.apply_transformation(cleaved_cell)
    return cleaved_cell

def get_unique_values_in_nth_value(arr_list, n, tolerance):
    """
    Returns unique values in the n-th element of sublists in arr_list within a specified tolerance.

    Parameters:
    - arr_list (list): List of sublists.
    - n (int): Index of the element to consider in each sublist.
    - tolerance (float): Tolerance for considering values as equal.

    Returns:
    - numpy.ndarray: Sorted array of unique values within the specified tolerance.
    """
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
    """
    Computes the average of consecutive pairs in the given list.

    Parameters:
    - lst (list): List of numerical values.

    Returns:
    - list: List of computed averages for consecutive pairs.
    """
    averages = []
    for i in range(len(lst) - 1):
        average = (lst[i] + lst[i + 1]) / 2
        averages.append(average)
    return averages

def get_non_host_ele_idx(structure, host_elements):
    """
    Returns the indices of non-host elements in the structure.

    Parameters:
    - structure (pymatgen.Structure): Structure object.
    - host_elements (list): List of host elements.

    Returns:
    - list: Indices of non-host elements in the structure.
    """
    non_host_indices = [i for i, site in enumerate(structure) if site.species_string not in host_elements]
    return non_host_indices

def get_min_max_cp_coords_solute(structure, host_elements, axis, threshold=5, fractional=True):
    """
    Returns the minimum and maximum coordinates of solute elements along the specified axis.

    Parameters:
    - structure (pymatgen.Structure): Structure object.
    - host_elements (list): List of host elements.
    - axis (int): Axis along which to determine the coordinates.
    - threshold (float): Threshold for determining the range.
    - fractional (bool): If True, returns fractional coordinates; otherwise, returns absolute coordinates.

    Returns:
    - list: [min_coord, max_coord] representing the range of coordinates.
    """
    non_host_indices = get_non_host_ele_idx(structure, host_elements)
    max_coord = None
    min_coord = None
    for site_idx in non_host_indices:
        coord = structure[site_idx].frac_coords[axis]
        if max_coord is None or coord > max_coord:
            max_coord = (coord + threshold/structure.lattice.abc[axis]) 
        if min_coord is None or coord < min_coord:
            min_coord = (coord - threshold/structure.lattice.abc[axis])
    if not fractional:
        max_coord = max_coord * structure.lattice.abc[axis]
        min_coord = min_coord * structure.lattice.abc[axis]
    return [min_coord, max_coord]

def get_cp_coords_solute(structure, host_elements, axis, threshold=5, tolerance=0.01, fractional=True):
    """
    Returns viable coordinates for solute elements within a specified range along the specified axis.

    Parameters:
    - structure (pymatgen.Structure): Structure object.
    - host_elements (list): List of host elements.
    - axis (int): Axis along which to determine the coordinates.
    - threshold (float): Threshold for determining the range.
    - tolerance (float): Tolerance for considering values as equal.
    - fractional (bool): If True, returns fractional coordinates; otherwise, returns absolute coordinates.

    Returns:
    - list: List of viable coordinates for solute elements.
    """
    min_max = get_min_max_cp_coords_solute(structure, host_elements, axis, fractional=fractional, threshold=threshold)
    if fractional:
        atomic_layers = get_unique_values_in_nth_value(structure.frac_coords, -1, tolerance=tolerance)
    else:
        atomic_layers = get_unique_values_in_nth_value(structure.cart_coords, -1, tolerance=tolerance)
    cp_list = compute_average_pairs(atomic_layers)
    min_cp_thres = min_max[0]
    max_cp_thres = min_max[1]
    # Count the number of floats in cp_list that are between min_cp_thres and max_cp_thres
    cp_viable = [cp for cp in cp_list if min_cp_thres <= cp <= max_cp_thres]
    return cp_viable

def cleave_structure(structure, cleave_line_coord, cleave_vacuum_length, axis, fractional=True):
    """
    Cleaves the structure along a specified coordinate line. 
    Assumes vacuum is already present! 
    If not, please structure = add_vacuum(structure) before this!

    Parameters:
    - structure (pymatgen.Structure): Structure object.
    - cleave_line_coord (float): Coordinate along which to cleave the structure.
    - vacuum_size (float): Size of vacuum to create between the two cleaved structures.
    - axis (int): Axis along which to cleave the structure.
    - fractional (bool): If True, uses fractional coordinates; otherwise, uses absolute coordinates.

    Returns:
    - pymatgen.Structure: Cleaved structure.
    """
    site_list = []; site_list2 = []
    for idx, sites in enumerate(structure):
        if fractional:
            if sites.frac_coords[axis] > cleave_line_coord:     
                site_list.append(idx)
            else:
                site_list2.append(idx)
        else:
            if sites.coords[axis] > cleave_line_coord:     
                site_list.append(idx)
            else:
                site_list2.append(idx)
    shift = [0, 0, 0]
    shift[axis] = cleave_vacuum_length/2
    shift2 = shift.copy()
    shift2[axis] = -cleave_vacuum_length/2
    transformation_shift_up = transform.TranslateSitesTransformation(site_list,tuple(shift),vector_in_frac_coords=False)
    transformation_shift_down = transform.TranslateSitesTransformation(site_list2,tuple(shift2),vector_in_frac_coords=False)
    cleaved_struct = transformation_shift_up.apply_transformation(structure)
    cleaved_struct = transformation_shift_down.apply_transformation(cleaved_struct)
    return cleaved_struct

def cleave_structure_around_solutes(structure,
                                    host_elements,
                                    axis=2,
                                    cleave_vacuum_length=6,
                                    sol_dist_threshold=5,
                                    tolerance=0.01,
                                    add_vacuum_block_length=None):
    if add_vacuum_block_length is not None:
        structure = add_vacuum(structure,vacuum=add_vacuum_block_length)
    cp_coords = get_cp_coords_solute(structure, host_elements=host_elements, axis=axis, threshold=sol_dist_threshold, tolerance=tolerance)
    cleaved_struct_list = []
    for cp in cp_coords:
        cleaved_struct = cleave_structure(structure,cleave_line_coord=cp,cleave_vacuum_length=cleave_vacuum_length, axis=axis)
        cleaved_struct_list.append(cleaved_struct)
    return cleaved_struct_list