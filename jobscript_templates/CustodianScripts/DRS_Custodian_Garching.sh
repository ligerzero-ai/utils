#!/bin/bash
#SBATCH --partition=p.cmfe
#SBATCH --ntasks={CPUSTRING}
#SBATCH --constraint='[swi1|swi1|swi2|swi3|swi4|swi5|swi6|swi7|swi8|swi9]'
#SBATCH --time={WALLTIMESTRING}
#SBATCH --output=time.out
#SBATCH --error=error.out
#SBATCH --job-name={CASESTRING}
#SBATCH --get-user-env=L

module purge

module load intel/19.1.0 impi/2019.6
module load vasp/5.4.4-buildFeb20

/cmmc/ptmp/hmai/mambaforge/bin/activate pymatgen
echo '{CUSTODIANSTRING}'>custodian_vasp.py

python custodian_vasp.py srun -n {CPUSTRING} --exclusive --mem-per-cpu=0 -m block:block,Pack vasp_std &> vasp.log

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
Chargemol_09_26_2017_linux_parallel

# Cleanup the data so it doesn't flood the drive
find . -type f \( -name "CHG.*" -o -name "WAVECAR*" -o -name "PROCAR*" -o -name "IBZKPT*" -o -name "REPORT*" -o -name "EIGENVAL*" -o -name "AECCAR*" -o -name "DOSCAR.*" \) -delete

