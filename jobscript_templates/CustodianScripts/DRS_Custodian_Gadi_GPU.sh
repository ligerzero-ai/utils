#!/bin/bash
#PBS -l walltime={WALLTIMESTRING}
#PBS -l mem={MEMORYSTRING}
#PBS -l ncpus={CPUSTRING}
#PBS -l software=vasp
#PBS -l ngpus={GPUSTRING}
#PBS -l wd
#PBS -l jobfs=1GB
#PBS -q gpuvolta
#PBS -lstorage=scratch/v43+gdata/v43

cd "$PBS_O_WORKDIR"

module load vasp/6.3.2

source /g/data/v43/Han/mambaforge/bin/activate pyiron

run_cmd="mpirun -np $PBS_NGPUS --map-by ppr:1:numa"

echo '{CUSTODIANSTRING}'>custodian_vasp.py

python custodian_vasp.py $run_cmd vasp_std-gpu &> vasp.log

echo '<net charge>
0.0 <-- specifies the net charge of the unit cell (defaults to 0.0 if nothing specified)
</net charge>
<periodicity along A, B, and C vectors>
.true. <--- specifies whether the first direction is periodic
.true. <--- specifies whether the second direction is periodic
.true. <--- specifies whether the third direction is periodic
</periodicity along A, B, and C vectors>
<atomic densities directory complete path>
/home/562/hlm562/chargemol_09_26_2017/atomic_densities/
</atomic densities directory complete path>
<charge type>
DDEC6 <-- specifies the charge type (DDEC3 or DDEC6)
</charge type>
<compute BOs>
.true. <-- specifies whether to compute bond orders or not
</compute BOs>' > job_control.txt

OMP_NUM_THREADS=48
export OMP_NUM_THREADS
export PATH=$PATH:/home/562/hlm562/chargemol_09_26_2017/atomic_densities/
export PATH=$PATH:/home/562/hlm562/chargemol_09_26_2017/chargemol_FORTRAN_09_26_2017/compiled_binaries/linux
Chargemol_09_26_2017_linux_parallel

# Cleanup the data so it doesn't flood the drive
rm CHG* CHGCAR* PROCAR* WAVECAR* EIGENVAL* REPORT* IBZKPT* REPORT* DOSCAR.* XDATCAR*

