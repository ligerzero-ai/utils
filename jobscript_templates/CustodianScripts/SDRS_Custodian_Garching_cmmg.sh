#!/bin/bash
#SBATCH --partition=s.cmmg
#SBATCH --ntasks={CPUSTRING}  # Adjust CPU count as needed
#SBATCH --cpus-per-task=1
#SBATCH --time={WALLTIMESTRING}  # Adjust wall time as needed
#SBATCH --output=time.out
#SBATCH --error=error.out
#SBATCH --job-name={CASESTRING}  # Adjust job name as needed
#SBATCH --get-user-env=L
#SBATCH --mem-per-cpu=2000MB
#SBATCH --hint=nomultithread
##SBATCH --reservation=benchmarking

module purge

module load intel/2024.0
module load impi/2021.11
module load mkl/2024.0

/cmmc/ptmp/hmai/mambaforge/bin/activate pymatgen

echo '{CUSTODIANSTRING}'>custodian_vasp.py

python custodian_vasp.py srun -c 1 -n {CPUSTRING} --hint=nomultithread /cmmc/ptmp/hmai/vasp_compiled/intel_suite/vasp.6.4.3_intelsuite_march_znver4/bin/vasp_std >> vasp.log

# Cleanup the data so it doesn't flood the drive
rm AECCAR* CHG* CHGCAR* PROCAR* WAVECAR* EIGENVAL* REPORT* IBZKPT* REPORT* DOSCAR.*