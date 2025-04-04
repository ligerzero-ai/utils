import sys
from custodian.custodian import Custodian
from custodian.vasp.handlers import (
    VaspErrorHandler,
    NonConvergingErrorHandler,
    PositiveEnergyErrorHandler,
    FrozenJobErrorHandler,
)
from utils.custom_custodian_handlers import Han_CustomVaspErrorHandler
from custodian.vasp.jobs import VaspJob

output_filename = {VASPOUTPUTFILENAME}

handlers = [
    VaspErrorHandler(output_filename=output_filename),
    Han_CustomVaspErrorHandler(),
    NonConvergingErrorHandler(),
    PositiveEnergyErrorHandler(),
    FrozenJobErrorHandler(output_filename=output_filename),
]

# Original job list
original_jobs = [
    VaspJob(
        sys.argv[1:],
        output_file=output_filename,
        suffix=".relax_1",
        final=False,
        settings_override=[
            {"file": "KPOINTS", "action": {"_file_move": {"dest": "KPOINTS_moved"}}},
            {
                "dict": "INCAR",
                "action": {
                    "_set": {
                        "NSW": 100,
                        "LAECHG": False,
                        "LCHARGE": False,
                        "NELM": 80,
                        "EDIFF": 1e-5,
                        "KSPACING": 0.9,
                    }
                },
            },
        ],
    ),
    VaspJob(
        sys.argv[1:],
        output_file=output_filename,
        suffix=".relax_1",
        final=False,
        settings_override=[
            {"file": "CHGCAR", "action": {"_file_copy": {"dest": "CHGCAR.static_1"}}},
            {"file": "AECCAR0", "action": {"_file_copy": {"dest": "AECCAR0.static_1"}}},
            {"file": "AECCAR1", "action": {"_file_copy": {"dest": "AECCAR1.static_1"}}},
            {"file": "AECCAR2", "action": {"_file_copy": {"dest": "AECCAR2.static_1"}}},
            {
                "dict": "INCAR",
                "action": {
                    "_set": {
                        "NSW": 100,
                        "LAECHG": False,
                        "LCHARGE": False,
                        "NELM": 80,
                        "EDIFF": 1e-4,
                        "KSPACING": 0.9,
                    }
                },
            },
        ],
        copy_magmom=True,
    ),
    VaspJob(
        sys.argv[1:],
        output_file=output_filename,
        suffix=".relax_2",
        final=False,
        settings_override=[
            {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}},
            {"dict": "INCAR", "action": {"_set": {"KSPACING": 0.5, "EDIFF": 1e-5}}},
        ],
        copy_magmom=True,
    ),
    VaspJob(
        sys.argv[1:],
        output_file=output_filename,
        suffix="",
        settings_override=[
            {
                "dict": "INCAR",
                "action": {
                    "_set": {
                        "NSW": 0,
                        "LAECHG": True,
                        "LCHARGE": True,
                        "NELM": 500,
                        "ALGO": "VeryFast",
                    }
                },
            }
        ],
    ),
]
# Number of elements to get from the end of the list
n = {STAGE}

# Extract the last n elements from the job list
jobs = original_jobs[-n:]

c = Custodian(handlers, jobs, max_errors=15)
c.run()
