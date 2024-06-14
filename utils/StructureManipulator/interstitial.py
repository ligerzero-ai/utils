import os
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import time

from pymatgen.core import Structure
import pymatgen.transformations.site_transformations as transform
from pymatgen.transformations.standard_transformations import SubstitutionTransformation
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core.lattice import Lattice

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Potcar, Incar, Kpoints
from pymatgen.io.vasp.outputs import Outcar
from pymatgen.core import Element

import pyscal.core as pc

from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import scipy.optimize as optimization

import json

# Reload packages when they change (mostly for custom modules)
# from IPython.lib.deepreload import reload
# %load_ext autoreload
# %autoreload 2

import warnings

warnings.filterwarnings("ignore")


# pyscal version <= 2.10.15
def get_all_vertices(sys):
    """
    Calculate all Voronoi vertices

    Parameters
    ----------
    sys: pyscal System object

    Returns
    -------
    all_vertices_raw: list of floats
    list of all Voronoi vertices
    """
    sys.find_neighbors(method="voronoi")
    all_vertices_raw = []
    for atom in sys.iter_atoms():
        for v in atom.vertex_positions:
            all_vertices_raw.append(v)
    return all_vertices_raw


def get_octahedral_positions(sys_in, alat):
    """
    Get all octahedral vertex positions

    Parameters
    ----------
    sys_in: pyscal System object

    alat: float
    lattice constant in Angstroms

    Returns
    -------
    octahedral_at: list of floats
    position of octahedral voids
    """
    octahedral_at = []
    real_pos = np.array([np.array(atox.pos) for atox in sys_in.iter_atoms()])
    atoms = sys_in.get_all_atoms()
    box = sys_in.box
    count = 0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dist = sys_in.get_distance(atoms[i], atoms[j])
            if np.abs(dist - alat) < 1e-2:
                count += 1
                npos = (np.array(atoms[i].pos) + np.array(atoms[j].pos)) / 2
                if 0 <= npos[0] <= box[0][0]:
                    if 0 <= npos[1] <= box[1][1]:
                        if 0 <= npos[2] <= box[2][2]:
                            # print(np.abs(np.sum(npos-real_pos)))
                            # print(npos)
                            found = False
                            for rpos in real_pos:
                                if np.sum(np.abs(npos - rpos)) < 1e-5:
                                    found = True
                            if not found:
                                octahedral_at.append(npos)
    return octahedral_at


def add_sphereatoms(sys, all_vertices, max_type):
    """
    Add ghost atoms at vertex positions

    Parameters

    ----------
    sys: pyscal System object

    all_vertices: list of floats
    list of vertex positions

    max_type: int
    maximum of all type numbers assigned to species

    Returns
    -------
    sys: pyscal System object

    """
    new_atoms = []
    for vertex in all_vertices:
        atom = pc.Atom(pos=vertex, type=max_type + 1)
        new_atoms.append(atom)
    sys.add_atoms(new_atoms)
    return sys


def get_ra(sys, natoms, pf):
    """
    Calculate radius ra

    Parameters
    ----------
    sys: pyscal System object

    natoms: int
    total number of atoms in the system

    pf: float
    packing factor of the system

    Returns
    -------
    ra: float
    Calculated ra
    """
    box = sys.box
    vol = np.dot(np.cross(box[0], box[1]), box[2])
    volatom = vol / natoms
    ra = ((pf * volatom) / ((4 / 3) * np.pi)) ** (1 / 3)
    return ra


def get_rvv(sys, max_type, ra):
    """
    Calculate rvv for each atom

    Parameters
    ----------
    sys: pyscal System object

    max_type: int
    maximum of all type numbers assigned to species

    ra: float
    calculated ra value
    """
    rlist = []
    atoms = sys.atoms
    for atom in atoms:
        if atom.type == max_type + 1:
            # collect ”real” neighbors
            nns = [x for x in atom.neighbors if atoms[x].type <= max_type]
            # get the distances
            dists = [sys.get_distance(atom, atoms[n]) for n in nns]
            # get minimum distance
            Rvv = min(dists)
            rvv = (Rvv - ra) / ra
            atom.cutoff = rvv
            rlist.append(rvv)
    return rlist, atoms


def get_interstitial_structure(
    input_file, output_file="poscar.vasp", alat=2.84, pf=0.68
):
    # Input parameters
    """
    pf = 0.68 # Packing factor of the input crystal lattice
    alat = 2.84 # Lattice constant in Angstroms
    example usage:
    iGB_struct = get_interstitial_structure("tempGB.vasp", output_file = "GB.vasp", alat = 2.84, pf = 0.68)
    struct_list, struct_all_studied_sites = get_int_struct_list(GB_struct_list[i], midpoint=midpoints[i])

    """
    # Read input from CONTCAR file
    sys_in = pc.System()
    sys_out = pc.System()

    sys_in.read_inputfile(input_file, format="poscar")
    sys_out.read_inputfile(input_file, format="poscar")

    # Find all Voronoi vertices and obtain unique ones with a precision of 2 decimal points
    all_vertices_raw = get_all_vertices(sys_in)
    all_vertices = np.unique(
        (np.array(all_vertices_raw) * 100).astype(int) / 100, axis=0
    )

    # Get all octahedral positions
    octahedral_at = get_octahedral_positions(sys_in, alat)

    # Get all types of atoms in the system
    conc = sys_in.get_concentration()
    natoms = sys_in.natoms
    max_type = len(conc.keys())

    # Combine vertices and octahedral sites
    combined_list = np.concatenate((all_vertices, octahedral_at))

    # add ghost atoms at vertex positions
    sys_out = add_sphereatoms(sys_out, combined_list, max_type)

    # calculate ra
    ra = get_ra(sys_out, natoms, pf)

    # Ghost atoms are used in pyscal to compensate for the small number of total real atoms
    # The remap_atoms method removes these ghost atoms, including:
    # (i) remapping atoms back to the simulation box,
    # (ii) remove the pyscal inbuilt ghost atoms, given by atom.id > total atoms,
    # (iii) remove atoms that are too close to each other - the distance tolerance can be set using ‘dtol‘

    nx = sys_out.remap_atoms(dtol=0.4, remove_images=False)
    sys_out.to_file(output_file, format="poscar")

    struct = Structure.from_file(filename=output_file)
    # Define a mapping from the current species to the desired species
    species_mapping = {Element("H"): Element("Fe"), Element("He"): Element("H")}
    # Create a substitution transformation to change the species
    substitution = SubstitutionTransformation(species_mapping)
    # Apply the substitution transformation to the structure
    struct = substitution.apply_transformation(struct)
    return struct
    # Get radius of VV sphere, rvv
    # Calculate neighbors again


#     sys_out.find_neighbors(method="cutoff",cutoff=alat)

#     # Output void ratios (rvv/ra) and count
#     rlist,atoms = get_rvv(sys_out,max_type,ra)
#     void_ratios, void_count = np.unique(np.round(rlist, decimals=1), return_counts=True)


def get_int_struct_list(
    structure, zlims=[], host_elements=["Fe"], within_GB_distance=3, midpoint=0.50945
):

    int_id = [
        i
        for i, site in enumerate(structure)
        if site.species_string not in host_elements
    ]
    GB_id = [
        i for i, site in enumerate(structure) if site.species_string in host_elements
    ]

    GB_struct = structure.copy()
    GB_struct.remove_sites(int_id)

    z_frac_coords = [site.frac_coords[-1] for site in GB_struct]
    zlims = [min(z_frac_coords), max(z_frac_coords)]

    only_intsites_struct = structure.copy()
    only_intsites_struct.remove_sites(GB_id)
    only_intsites_struct.merge_sites(tol=0.35, mode="a")
    only_intsites_struct.sort(lambda x: x.frac_coords[-1])

    int_fcoords = [
        site.frac_coords for site in only_intsites_struct if site.species_string == "H"
    ]

    # Get the ones we are interested in computing
    struct_list = []
    struct_all_studied_sites = GB_struct.copy()

    for int_sites in int_fcoords:
        compute_struct = GB_struct.copy()
        if (
            int_sites[-1] > min(zlims)
            and int_sites[-1] < max(zlims)
            and abs(int_sites[-1] - midpoint) * structure.lattice.c < within_GB_distance
        ):
            # and int_sites[-1] <= midpoint+0.01:
            compute_struct.append("H", int_sites)
            struct_list.append(compute_struct)
            struct_all_studied_sites.append("H", int_sites)

    return struct_list, struct_all_studied_sites
