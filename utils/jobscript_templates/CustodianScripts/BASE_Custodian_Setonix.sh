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