#!/bin/bash -l
#SBATCH --ntasks={CPUSTRING}
#SBATCH --ntasks-per-node={CPUSTRING}
#SBATCH --cpus-per-task=1
#SBATCH --account=pawsey0380
#SBATCH --job-name={CASESTRING}
#SBATCH --time={WALLTIMESTRING}
#SBATCH --partition=work
#SBATCH --export=NONE
#SBATCH --mem-per-cpu=1500MB
##SBATCH --exclusive

module load vasp/5.4.4
cd "$PBS_O_WORKDIR"

ulimit -s unlimited
run_cmd="srun --export=ALL -N 1 -n {CPUSTRING}"

source /scratch/pawsey0380/hmai/mambaforge/bin/activate pymatgen

# Measure and print the runtime of vasp_std
start_time=$(date +%s)
$run_cmd vasp_std
end_time=$(date +%s)

runtime=$((end_time - start_time))
echo "Runtime of vasp_std: ${runtime} seconds" > consumed_time.txt
module load vasp/5.4.4
cd "$PBS_O_WORKDIR"

ulimit -s unlimited
run_cmd="srun --export=ALL -N {NODESTRING} -n {CPUSTRING}"

source /scratch/pawsey0380/hmai/mambaforge/bin/activate pymatgen

echo 'import sys

from custodian.custodian import Custodian
from custodian.vasp.jobs import VaspJob
from custodian.vasp.handlers import (
    VaspErrorHandler, 
    NonConvergingErrorHandler,
#    PositiveEnergyErrorHandler,
    FrozenJobErrorHandler
)
from utils.custom_custodian_handlers import Han_CustomVaspErrorHandler

output_filename = {VASPOUTPUTFILENAME}

handlers = [
    VaspErrorHandler(output_filename=output_filename),
    Han_CustomVaspErrorHandler(),
    NonConvergingErrorHandler(),
 #   PositiveEnergyErrorHandler(),
    FrozenJobErrorHandler(output_filename=output_filename)
]
output_filename = "vasp.log"
jobs = [VaspJob(sys.argv[1:],
                output_file=output_filename,
                suffix = "")]
c = Custodian(handlers, jobs, max_errors=10)
c.run()'>custodian_vasp.py

python custodian_vasp.py $run_cmd vasp_std &> vasp.log

# Cleanup the data so it doesn't flood the drive
rm CHG* CHGCAR* PROCAR* WAVECAR* EIGENVAL* REPORT* IBZKPT* REPORT* DOSCAR.* XDATCAR*
