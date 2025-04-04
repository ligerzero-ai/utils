#!/bin/bash
{%- if cores < 40 %}
#SBATCH --partition=s.cmfe
{%- else %}
#SBATCH --partition=p.cmfe
{%- endif %}
#SBATCH --ntasks={{cores}}
#SBATCH --constraint='[swi1|swi1|swi2|swi3|swi4|swi5|swi6|swi7|swi8|swi9]' 
{%- if run_time_max %}
#SBATCH --time={{ [1, run_time_max // 60]|max }}
{%- endif %}
{%- if memory_max %}
#SBATCH --mem={{memory_max}}G
{%- else %}
{%- if cores < 40 %}
#SBATCH --mem-per-cpu=3GB
{%- endif %}
{%- endif %}
#SBATCH --output=time.out
#SBATCH --error=error.out
#SBATCH --job-name={{job_name}}
#SBATCH --chdir={{working_directory}}
#SBATCH --get-user-env=L
#SBATCH --qos=longrun -t 29-0

pwd; 
echo Hostname: `hostname`
echo Date: `date`
echo JobID: $SLURM_JOB_ID

{{command}}
