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

echo '{CUSTODIANSTRING}'>custodian_vasp.py

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
</compute BOs>'>job_control.txt

OMP_NUM_THREADS={CPUSTRING}
export OMP_NUM_THREADS
export PATH=$PATH:/home/hmai/chargemol_09_26_2017/atomic_densities/
export PATH=$PATH:/home/hmai/chargemol_09_26_2017/chargemol_FORTRAN_09_26_2017/compiled_binaries/linux

current_dir=$(pwd)

mkdir DDEC6_relaxed
mv AECCAR0 AECCAR1 AECCAR2 CHGCAR DDEC6_relaxed/
cp job_control.txt POTCAR DDEC6_relaxed/
cd DDEC6_relaxed
$run_cmd Chargemol_09_26_2017_linux_parallel
cd "$current_dir"

mkdir DDEC6_initial
mv AECCAR0.static_1 DDEC6_initial/AECCAR0
mv AECCAR1.static_1 DDEC6_initial/AECCAR1
mv AECCAR2.static_1 DDEC6_initial/AECCAR2
mv CHGCAR.static_1 DDEC6_initial/CHGCAR
cp job_control.txt POTCAR DDEC6_initial/
cd DDEC6_initial
$run_cmd Chargemol_09_26_2017_linux_parallel
cd "$current_dir"

# Cleanup the data so it doesn't flood the drive
#rm CHG* CHGCAR* PROCAR* WAVECAR* EIGENVAL* REPORT* IBZKPT* REPORT* DOSCAR.relax_1 DOSCAR.relax_2
find . -type f \( -name "CHG*" -o -name "WAVECAR*" -o -name "PROCAR*" -o -name "IBZKPT*" -o -name "REPORT*" -o -name "EIGENVAL*" -o -name "AECCAR*" -o -name "DOSCAR.*" \) -delete