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

# Cleanup the data so it doesn't flood the drive
rm AECCAR* CHG* CHGCAR* PROCAR* WAVECAR* EIGENVAL* REPORT* IBZKPT* REPORT* DOSCAR.*