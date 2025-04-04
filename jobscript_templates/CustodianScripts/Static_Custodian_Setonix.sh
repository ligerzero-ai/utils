#!/bin/bash -l
##SBATCH --nodes={NODESTRING}
#SBATCH --ntasks={CPUSTRING}
#SBATCH --account=pawsey0380
#SBATCH --job-name={CASESTRING}
#SBATCH --time={WALLTIMESTRING}
#SBATCH --partition=work
#SBATCH --export=NONE
##SBATCH --exclusive
#SBATCH --mem=32GB
module load vasp/5.4.4
cd "$PBS_O_WORKDIR"

ulimit -s unlimited
run_cmd="srun --export=ALL -N {NODESTRING} -n {CPUSTRING}"

source /scratch/pawsey0380/hmai/mambaforge/bin/activate pymatgen

echo '{CUSTODIANSTRING}'>custodian.py

python custodian.py $run_cmd vasp_std &> vasp.log

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
# directory_name=$(basename "$PWD")
# tar -czvf "${directory_name}.tar.gz" --exclude="AECCAR*" .
# find . ! -name "${directory_name}.tar.gz" ! -name 'OUTCAR' ! -name 'CONTCAR' ! -name 'vasprun.xml' ! -name 'starter*' ! -name 'AECCAR*' -type f -exec rm -f {} +
find . -type f \( -name "WAVECAR*" -o -name "PROCAR*" -o -name "IBZKPT*" -o -name "REPORT*" -o -name "EIGENVAL*" \) -delete
find . -type f \( -name "CHG*" -o -name "WAVECAR*" -o -name "PROCAR*" -o -name "IBZKPT*" -o -name "REPORT*" -o -name "EIGENVAL*" -o -name "AECCAR*" -o -name "DOSCAR.*" \) -delete