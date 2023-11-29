from pymatgen.core import Structure
import pymatgen.transformations.site_transformations as transform
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core.lattice import Lattice

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