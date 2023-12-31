#!/bin/bash
#SBATCH --partition=p.cmfe
#SBATCH --ntasks={CPUSTRING}
#SBATCH --constraint='[swi1|swi1|swi2|swi3|swi4|swi5|swi6|swi7|swi8|swi9|swe1|swe2|swe3|swe4|swe5|swe6|swe7]'
#SBATCH --time={WALLTIMESTRING}
#SBATCH --output=time.out
#SBATCH --error=error.out
#SBATCH --job-name={CASESTRING}
#SBATCH --get-user-env=L

module load intel/19.1.0 impi/2019.6 
module load vasp/5.4.4-buildFeb20 
module load pyiron/dev 

echo 'import sys

from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler, NonConvergingErrorHandler, PositiveEnergyErrorHandler, FrozenJobErrorHandler
from utils.custom_custodian_handlers import Han_CustomVaspErrorHandler

from custodian.vasp.jobs import VaspJob

output_filename = "vasp.log"
handlers = [VaspErrorHandler(output_filename=output_filename), Han_CustomVaspErrorHandler(), NonConvergingErrorHandler(), PositiveEnergyErrorHandler(), FrozenJobErrorHandler(output_filename=output_filename)]
jobs = [VaspJob(sys.argv[1:], output_file=output_filename, suffix = ".relax_1", final=False, settings_override=[{"dict": "INCAR", "action": {"_set": {"KSPACING": 0.9}}}]),
        VaspJob(sys.argv[1:], output_file=output_filename, suffix = ".relax_2", final=False,
                settings_override = [{"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}},
                    {"dict": "INCAR", "action": {"_set": {"KSPACING": 0.5, "EDIFF": 1E-5, "EDIFFG": -0.01}}}], copy_magmom=True),
        VaspJob(sys.argv[1:], output_file=output_filename, suffix = "",
                        settings_override = [{"dict": "INCAR",
                              "action": {
                                  "_set":
                                  {"NSW": 0, "LAECHG": True, "LCHARGE": True, "NELM": 500, "ALGO": "VeryFast", "EDIFF": 1E-5}
                                  }
                              }])]
c = Custodian(handlers, jobs, max_errors=15)
c.run()'>custodian_vasp.py

if [ $(hostname) == 'cmti001' ];
then
        unset I_MPI_HYDRA_BOOTSTRAP;
        unset I_MPI_PMI_LIBRARY;
        python custodian_vasp.py mpiexec -n $1 vasp_std
else
        python custodian_vasp.py srun -n {CPUSTRING} --exclusive --mem-per-cpu=0 -m block:block,Pack vasp_std &> vasp.log
fi

echo '<net charge>
0.0 <-- specifies the net charge of the unit cell (defaults to 0.0 if nothing specified)
</net charge>
<periodicity along A, B, and C vectors>
.true. <--- specifies whether the first direction is periodic
.true. <--- specifies whether the second direction is periodic
.true. <--- specifies whether the third direction is periodic
</periodicity along A, B, and C vectors>
<atomic densities directory complete path>
/cmmc/u/hmai/chargemol_09_26_2017/atomic_densities/
</atomic densities directory complete path>
<charge type>
DDEC6 <-- specifies the charge type (DDEC3 or DDEC6)
</charge type>
<compute BOs>
.true. <-- specifies whether to compute bond orders or not
</compute BOs>'>job_control.txt

OMP_NUM_THREADS={CPUSTRING}
export OMP_NUM_THREADS
export PATH=$PATH:/cmmc/u/hmai/chargemol_09_26_2017/atomic_densities/
export PATH=$PATH:/cmmc/u/hmai/chargemol_09_26_2017/chargemol_FORTRAN_09_26_2017/compiled_binaries/linux
srun --export=ALL -N {NODESTRING} -n {CPUSTRING} Chargemol_09_26_2017_linux_parallel

# Cleanup the data so it doesn't flood the drive
find . -type f \( -name "CHG.*" -o -name "WAVECAR*" -o -name "PROCAR*" -o -name "IBZKPT*" -o -name "REPORT*" -o -name "EIGENVAL*" -o -name "AECCAR*" -o -name "DOSCAR.*" \) -delete

