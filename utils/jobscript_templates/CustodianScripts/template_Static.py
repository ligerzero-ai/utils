import sys
from custodian.custodian import Custodian
from custodian.vasp.handlers import (
    VaspErrorHandler, 
    NonConvergingErrorHandler,
    PositiveEnergyErrorHandler, 
    FrozenJobErrorHandler
)
from utils.custom_custodian_handlers import Han_CustomVaspErrorHandler
from custodian.vasp.jobs import VaspJob

output_filename = {VASPOUTPUTFILENAME}

handlers = [
    VaspErrorHandler(output_filename=output_filename),
    Han_CustomVaspErrorHandler(),
    NonConvergingErrorHandler(),
    PositiveEnergyErrorHandler(),
    FrozenJobErrorHandler(output_filename=output_filename)
]

jobs = [VaspJob(sys.argv[1:], output_file=output_filename, suffix = "",
                        settings_override = [{"dict": "INCAR",
                                              "action": {"_set":{"NSW": 0, "LAECHG": True, "LCHARGE": True, "NELM": 500, "ALGO": "VeryFast", "EDIFF": 1E-5}}}]
                )]
c = Custodian(handlers, jobs, max_errors={MAXCUSTODIANERRORS})

c.run()
