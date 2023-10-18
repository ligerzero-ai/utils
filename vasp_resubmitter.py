import os
import shutil
import tarfile
import subprocess
import pandas as pd

from utils.vasp import find_vasp_directories, check_convergence
from utils.generic import get_latest_file_iteration

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
        
    def reconverge_all(self):
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
                    self.reconverge(dir)
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
    
    def reconverge(self, dirpath):
        static1_files_exist = any(f.endswith(".static_1") for f in os.listdir(dirpath))
        relax1_files_exist = any(f.endswith(".relax_1") for f in os.listdir(dirpath))
        relax2_files_exist = any(f.endswith(".relax_2") for f in os.listdir(dirpath))

        # Check if .relax_1 and .relax2 files exist and use the static relaxation script
        if static1_files_exist:
            script_name = os.path.join(self.script_template_dir, "SDRS_Custodian_1.sh")
        if relax2_files_exist:
            script_name = os.path.join(self.script_template_dir, "SDRS_Custodian_2.sh")
        elif relax1_files_exist:
            script_name = os.path.join(self.script_template_dir, "SDRS_Custodian_3.sh")
        else:
            script_name = os.path.join(self.script_template_dir, "SDRS_Custodian.sh")
        
        target_script_name = f"{os.path.basename(dirpath)}.sh"
        
        shutil.copy(script_name, os.path.join(dirpath, target_script_name))
    
        self.submit_to_queue(dirpath, target_script_name)
    
    # def _get_latest_file_iteration(self, resubmit_log_filename = "resubmit.log_"):
    #     # Check for existing resubmit.log_m files and find the largest m
    #     resubmit_log_files = []
    #     for filename in os.listdir(self.parent_dir):
    #         if resubmit_log_filename in filename:
    #             resubmit_log_files.append(filename)
    #     max_integer = -1
    #     if not resubmit_log_files:
    #         return -1
    #     else:
    #         for log_file in resubmit_log_files:
    #             if log_file.startswith(resubmit_log_filename):
    #                 try:
    #                     num_str = log_file[len(resubmit_log_filename):]
    #                     num = int(num_str)
    #                     max_integer = max(max_integer, num)
    #                 except ValueError:
    #                     pass  # Ignore non-integer parts after "resubmit.log_"
    #         return max_integer
                
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
