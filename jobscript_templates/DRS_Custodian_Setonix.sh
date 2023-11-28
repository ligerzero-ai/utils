#!/bin/bash -l
#SBATCH --nodes={NODESTRING}
#SBATCH --ntasks={CPUSTRING}
#SBATCH --account=pawsey0380
#SBATCH --job-name={CASESTRING}
#SBATCH --time={WALLTIMESTRING}
#SBATCH --partition=work
#SBATCH --export=NONE
#SBATCH --exclusive

module load vasp/5.4.4
cd "$PBS_O_WORKDIR"

ulimit -s unlimited
run_cmd="srun --export=ALL -N {NODESTRING} -n {CPUSTRING}"

source /scratch/pawsey0380/hmai/mambaforge/bin/activate pymatgen

echo 'import sys

from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler, NonConvergingErrorHandler, PositiveEnergyErrorHandler
from custodian.vasp.jobs import VaspJob

output_filename = "vasp.log"
handlers = [VaspErrorHandler(output_filename=output_filename), UnconvergedErrorHandler(), NonConvergingErrorHandler(), PositiveEnergyErrorHandler()]
jobs = [VaspJob(sys.argv[1:],
                output_file=output_filename,
                suffix = ".relax_1", final=False,
                settings_override=[{"dict": "INCAR", "action": {"_set": {"NSW": 100, "LAECHG": False, "LCHARGE": False, "NELM": 100, "EDIFF": 1E-5, "KSPACING" : 0.9}}}]),
        VaspJob(sys.argv[1:],
                output_file=output_filename,
                suffix = ".relax_2",
                final=False,
                settings_override = [{"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}},
                                     {"dict": "INCAR", "action": {"_set": {"KSPACING": 0.5, "EDIFF": 1E-5, "EDIFFG": -0.01}}}],
                copy_magmom=True),
        VaspJob(sys.argv[1:],
                output_file=output_filename,
                suffix = "",
                settings_override = [{"dict": "INCAR", "action": {"_set": {"NSW": 0, "LAECHG": True, "LCHARGE": True, "NELM": 400, "EDIFF": 1E-5}}},
                                     {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}])]
c = Custodian(handlers, jobs, max_errors=10)
c.run()'>custodian_vasp.py

python custodian_vasp.py $run_cmd vasp_std &> vasp.log

echo '<net charge>
0.0 <-- specifies the net charge of the unit cell (defaults to 0.0 if nothing specified)
</net charge>
<periodicity along A, B, and C vectors>
.true. <--- specifies whether the first direction is periodic
.true. <--- specifies whether the second direction is periodic
.true. <--- specifies whether the third direction is periodic
</periodicity along A, B, and C vectors>
<atomic densities directory complete path>
/home/hmai/chargemol_09_26_2017/atomic_densities/
</atomic densities directory complete path>
<charge type>
DDEC6 <-- specifies the charge type (DDEC3 or DDEC6)
</charge type>
<compute BOs>
.true. <-- specifies whether to compute bond orders or not
</compute BOs>' > job_control.txt

OMP_NUM_THREADS={CPUSTRING}
export OMP_NUM_THREADS
export PATH=$PATH:/home/hmai/chargemol_09_26_2017/atomic_densities/
export PATH=$PATH:/home/hmai/chargemol_09_26_2017/chargemol_FORTRAN_09_26_2017/compiled_binaries/linux
$run_cmd Chargemol_09_26_2017_linux_parallel

# Cleanup the data so it doesn't flood the drive
find . -type f \( -name "CHG.*" -o -name "WAVECAR*" -o -name "PROCAR*" -o -name "IBZKPT*" -o -name "REPORT*" -o -name "EIGENVAL*" -o -name "AECCAR*" -o -name "DOSCAR.*" \) -delete
