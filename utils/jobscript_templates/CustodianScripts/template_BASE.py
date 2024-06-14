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

jobs = [VaspJob(sys.argv[1:],
                output_file=output_filename,
                suffix = "")]
c = Custodian(handlers, jobs, max_errors=10)
c.run()
