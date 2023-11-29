import os
import shutil
import tarfile
import subprocess
import pandas as pd

from utils.vasp import find_vasp_directories, check_convergence
from utils.generic import get_latest_file_iteration
from utils.jobfile import jobfile

def get_slurm_jobs_working_directories(username="hmai"):
    command = f"squeue -u {username} -o \"%i %Z\""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output_lines = result.stdout.strip().split("\n")[1:]  # Remove the header line
    
    # Parse the output lines into a list of tuples (job_id, working_directory)
    data = [line.split() for line in output_lines]
    
    # Create a Pandas DataFrame from the data
    df = pd.DataFrame(data, columns=["Job ID", "Working Directory"])
    
    return df

class CalculationConverger():
    
    def __init__(self, parent_dir, script_template_dir, max_submissions = 1000, submission_command = "sbatch", username="hmai"):
        self.parent_dir = parent_dir
        self.max_submissions = max_submissions
        self.submission_command = submission_command
        self.vasp_dirs = find_vasp_directories(parent_dir, filenames=["INCAR", "POTCAR"], all_present=True, extract_tarballs=False)
        self.script_template_dir = script_template_dir
        self.user = username

    def submit_to_queue(self, dirpath, script_name):
        os.system(f"cd {dirpath} && {self.submission_command} {script_name}")
        
    def reconverge_all(self,
                       calc_type = "DRS",
                       HPC = "Setonix",
                       VASP_version = "5.4.4",
                       CPU = 128,
                       walltime = 24,
                       cpu_per_node=128):
        non_converged = self.reconverge_from_log_file()
        running_jobs_df = get_slurm_jobs_working_directories(self.user)
        running_queued_job_directories = running_jobs_df["Working Directory"].to_list()
        
        dirs_to_search_next_time = []
        leftover_calcs_exceeding_queue_limit = []
        # Exclude the running job directories from dirs_to_search
        dirs_to_apply_reconverge = set(non_converged) if non_converged else set(self.vasp_dirs)
        dirs_to_apply_reconverge -= set(running_queued_job_directories)
        dirs_to_apply_reconverge = list(set(dirs_to_apply_reconverge))
        
        for i, dir in enumerate(dirs_to_apply_reconverge):
            converged = check_convergence(dir)
            if not converged:
                print(f"UNCONVERGED: {dir}")
                non_converged.append(dir)
                if i + len(running_queued_job_directories) > self.max_submissions:
                    leftover_calcs_exceeding_queue_limit.append(dir)
                else:
                    self.reconverge(dir,
                                    calc_type = calc_type,
                                    HPC = HPC,
                                    VASP_version = VASP_version,
                                    CPU = CPU,
                                    walltime = walltime,
                                    cpu_per_node=cpu_per_node)
                    dirs_to_search_next_time.append(dir)
            else:
                print(f"CONVERGED: {dir}")
        
        dirs_to_search_next_time += running_queued_job_directories
        dirs_to_search_next_time += leftover_calcs_exceeding_queue_limit
        
        os.chdir(self.parent_dir)
        
        with open(os.path.join(self.parent_dir, "resubmit.log"), "w") as log_file:
            for dir_path in dirs_to_search_next_time:
                log_file.write(dir_path + "\n")
                
        return dirs_to_search_next_time
    
    def reconverge(self,
                    dirpath,
                    calc_type="SDRS",
                    HPC = "Setonix",
                    VASP_version = "5.4.4",
                    CPU = 128,
                    walltime = 24,
                    cpu_per_node=128):

        # Check if there are any error* and *tar* files in the directory
        error_tar_files_exist = any("error" in f and "tar" in f for f in os.listdir(dirpath))

        # Create error_run_n folder if error* and *tar* files exist
        if error_tar_files_exist:
            # Create error_run_n folder
            latest_error_run_index = self.find_latest_error_run_index(dirpath)
            error_run_folder_name = f"error_run_{latest_error_run_index + 1}"
            error_run_folder_path = os.path.join(dirpath, error_run_folder_name)
            os.makedirs(error_run_folder_path)
            # Move files containing "error" and "tar" into the error_run_n folder
            for f in os.listdir(dirpath):
                if "error" in f and "tar" in f:
                    shutil.move(os.path.join(dirpath, f), os.path.join(error_run_folder_path, f))
                if f.endswith(".sh"):
                    shutil.move(os.path.join(dirpath, f), os.path.join(error_run_folder_path, f))
                    
            orig_files_to_preserve = ["INCAR.orig", "POSCAR.orig", "KPOINTS.orig", "custodian.json"]
            for og_file in orig_files_to_preserve:
                if os.path.exists(os.path.join(dirpath, og_file)):
                    shutil.move(os.path.join(dirpath, og_file), os.path.join(error_run_folder_path, og_file))
               
        if calc_type=="SDRS":
            self.reconverge_SDRS(dirpath,
                                HPC = HPC,
                                VASP_version = VASP_version,
                                CPU = CPU,
                                walltime = walltime,
                                cpu_per_node=cpu_per_node)
        elif calc_type=="DRS":
            self.reconverge_DRS(dirpath,
                                HPC = HPC,
                                VASP_version = VASP_version,
                                CPU = CPU,
                                walltime = walltime,
                                cpu_per_node=cpu_per_node)
            
    # Function to find the latest error_run folder index
    def find_latest_error_run_index(self, dirpath):
        error_run_indices = [0]
        for f in os.listdir(dirpath):
            if f.startswith("error_run_") and f[11:].isdigit():
                error_run_indices.append(int(f[11:]))
        return max(error_run_indices)    
    
    def reconverge_DRS(self,
                       dirpath,
                       HPC = "Setonix",
                       VASP_version = "5.4.4",
                       CPU = 128,
                       walltime = 24,
                       cpu_per_node=128
                       ):
        
        relax1_files_exist = any(f.endswith(".relax_1") for f in os.listdir(dirpath))
        relax2_files_exist = any(f.endswith(".relax_2") for f in os.listdir(dirpath))

        if relax2_files_exist:
            script_name = os.path.join(self.script_template_dir, f"DRS_Custodian_2_{HPC}.sh")
        elif relax1_files_exist:
            script_name = os.path.join(self.script_template_dir, f"DRS_Custodian_1_{HPC}.sh")
        else:
            script_name = os.path.join(self.script_template_dir, f"DRS_Custodian_{HPC}.sh")

        target_script_name = f"{os.path.basename(dirpath)}.sh"
        job = jobfile(file_path = script_name,
                    HPC = HPC,
                    VASP_version = VASP_version,
                    CPU = CPU,
                    walltime = walltime,
                    cpu_per_node=cpu_per_node)
        
        job.to_file(job_name=target_script_name,
                    output_path=dirpath)
        
        # Submit to the queue using the error_run_n folder
        self.submit_to_queue(dirpath, target_script_name)
        
    def reconverge_SDRS(self,
                        dirpath,
                        HPC = "Setonix",
                        VASP_version = "5.4.4",
                        CPU = 128,
                        walltime = 24,
                        cpu_per_node=128
                        ):
        static1_files_exist = any(f.endswith(".static_1") for f in os.listdir(dirpath))
        relax1_files_exist = any(f.endswith(".relax_1") for f in os.listdir(dirpath))
        relax2_files_exist = any(f.endswith(".relax_2") for f in os.listdir(dirpath))

        # Check if .relax_1 and .relax2 files exist and use the static relaxation script
        if static1_files_exist:
            script_name = os.path.join(self.script_template_dir, f"SDRS_Custodian_1_{HPC}.sh")
        if relax2_files_exist:
            script_name = os.path.join(self.script_template_dir, f"SDRS_Custodian_2_{HPC}.sh")
        elif relax1_files_exist:
            script_name = os.path.join(self.script_template_dir, f"SDRS_Custodian_3_{HPC}.sh")
        else:
            script_name = os.path.join(self.script_template_dir, f"SDRS_Custodian_{HPC}.sh")
        
        target_script_name = f"{os.path.basename(dirpath)}.sh"
        job = jobfile(file_path = script_name,
                    HPC = HPC,
                    VASP_version = VASP_version,
                    CPU = CPU,
                    walltime = walltime,
                    cpu_per_node=cpu_per_node)
        
        job.to_file(job_name=target_script_name,
                    output_path=dirpath)
            
        self.submit_to_queue(dirpath, target_script_name)
                 
    def reconverge_from_log_file(self):
        resubmit_log_file = os.path.join(self.parent_dir, "resubmit.log")        
        if os.path.isfile(resubmit_log_file):
            # Submit jobs from the resubmit_log_file
            with open(resubmit_log_file, "r") as log_file:
                non_converged_dirs = [line.strip() for line in log_file.readlines()]
            
            largest_n = get_latest_file_iteration(self.parent_dir, "resubmit.log_")
            # Rename the existing resubmit.log to resubmit.log_n
            new_log_filename = f"resubmit.log_{largest_n + 1}"
            os.rename(resubmit_log_file, new_log_filename)
            
            return non_converged_dirs
        else:
            print("No resubmit log file found. Nothing to resubmit from old logs.")
            return []
