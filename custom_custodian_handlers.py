"""
This module implements specific error handlers for VASP runs. These handlers
try to detect common errors in vasp runs and attempt to fix them on the fly
by modifying the input files.
"""

import datetime
import logging
import multiprocessing
import os
import re
import shutil
import time
import warnings
from collections import Counter
from math import prod

import numpy as np
from monty.os.path import zpath
from monty.serialization import loadfn
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, VaspInput
from pymatgen.io.vasp.outputs import Oszicar, Outcar, Vasprun
from pymatgen.io.vasp.sets import MPScanRelaxSet
from pymatgen.transformations.standard_transformations import SupercellTransformation

from custodian.ansible.actions import FileActions
from custodian.ansible.interpreter import Modder
from custodian.custodian import ErrorHandler
from custodian.utils import backup
from custodian.vasp.interpreter import VaspModder

__author__ = (
    "Han Lin Mai"
)
__version__ = "0.1"
__maintainer__ = "Han Mai"
__email__ = "h.mai@mpie.de"
__status__ = "Beta"
__date__ = "10 Dec 23"

VASP_BACKUP_FILES = {
    "INCAR",
    "KPOINTS",
    "POSCAR",
    "OUTCAR",
    "CONTCAR",
    "OSZICAR",
    "vasprun.xml",
    "vasp.out",
    "std_err.txt",
}

class Han_CustomVaspErrorHandler(ErrorHandler):
    """Check if a run is converged."""

    is_monitor = False

    def __init__(self, output_filename="vasprun.xml"):
        """
        Initializes the handler with the output file to check.

        Args:
            output_vasprun (str): Filename for the vasprun.xml file. Change
                this only if it is different from the default (unlikely).
        """
        self.output_filename = output_filename

    def check(self):
        """Check for error."""
        try:
            v = Vasprun(self.output_filename)
            if not v.converged:
                return True
        except Exception:
            pass
        return False

    def correct(self):
        """Perform corrections."""
        v = Vasprun(self.output_filename)
        algo = v.incar.get("ALGO", "Normal").lower()
        actions = []
        if not v.converged_electronic:

            # Ladder from VeryFast to Fast to Normal to All
            # (except for meta-GGAs and hybrids).
            # These progressively switch to more stable but more
            # expensive algorithms.
            if len(actions) == 0:
                if algo == "veryfast":
                    actions.append({"dict": "INCAR", "action": {"_set": {"ALGO": "Fast"}}})
                elif algo == "fast":
                    actions.append({"dict": "INCAR", "action": {"_set": {"ALGO": "Normal"}}})
                elif algo == "normal" and (v.incar.get("ISMEAR", -1) >= 0 or not 50 <= v.incar.get("IALGO", 38) <= 59):
                    actions.append({"dict": "INCAR", "action": {"_set": {"ALGO": "All"}}})
                else:
                    # Try mixing as last resort
                    new_settings = {
                        "ISTART": 1,
                        "ALGO": "Normal",
                        "NELMDL": -6,
                        "BMIX": 0.001,
                        "AMIX_MAG": 0.8,
                        "BMIX_MAG": 0.001,
                    }

                    if not all(v.incar.get(k, "") == val for k, val in new_settings.items()):
                        actions.append({"dict": "INCAR", "action": {"_set": new_settings}})

        elif not v.converged_ionic:
            # Just continue optimizing and let other handlers fix ionic
            # optimizer parameters
            actions += [
                {"dict": "INCAR", "action": {"_set": {"IBRION": 1}}},
                {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}},
            ]

        if actions:
            vi = VaspInput.from_directory(".")
            backup(VASP_BACKUP_FILES)
            VaspModder(vi=vi).apply_actions(actions)
            return {"errors": ["Unconverged"], "actions": actions}

        # Unfixable error. Just return None for actions.
        return {"errors": ["Unconverged"], "actions": None}