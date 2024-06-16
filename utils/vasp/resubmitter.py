import os
import shutil
import tarfile
import subprocess
import pandas as pd

from utils.vasp.database import find_vasp_directories, check_convergence
from utils.generic import get_latest_file_iteration
from utils.jobfile import jobfile


def get_slurm_jobs_working_directories(username="hmai"):
    command = f'squeue -u {username} -o "%i %Z"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output_lines = result.stdout.strip().split("\n")[1:]  # Remove the header line

    # Parse the output lines into a list of tuples (job_id, working_directory)
    data = [line.split() for line in output_lines]

    # Create a Pandas DataFrame from the data
    df = pd.DataFrame(data, columns=["Job ID", "Working Directory"])

    return df


class CalculationConverger:

    def __init__(
        self,
        parent_dir,
        script_template_dir,
        max_submissions=1000,
        submission_command="sbatch",
        username="hmai",
    ):
        self.parent_dir = parent_dir
        self.max_submissions = max_submissions
        self.submission_command = submission_command
        self.vasp_dirs = find_vasp_directories(
            parent_dir,
            filenames=["INCAR", "POTCAR"],
            all_present=True,
            extract_tarballs=False,
        )
        self.script_template_dir = script_template_dir
        self.user = username

    def submit_to_queue(self, dirpath, script_name):
        os.system(f"cd {dirpath} && {self.submission_command} {script_name}")

    def reconverge_all(
        self,
        calc_type="DRS",
        HPC="Setonix",
        VASP_version="5.4.4",
        CPU=128,
        walltime=24,
        cpu_per_node=128,
        from_dataframe_path=None,
    ):
        non_converged = self.load_non_converged_paths(from_dataframe_path)
        running_jobs_df = get_slurm_jobs_working_directories(self.user)
        running_queued_job_directories = running_jobs_df["Working Directory"].to_list()

        dirs_to_search_next_time, leftover_calcs_exceeding_queue_limit = [], []

        dirs_to_apply_reconverge = set(non_converged or self.vasp_dirs) - set(
            running_queued_job_directories
        )
        
        # Split directories into those without vasp.log and those with vasp.log
        dirs_without_log = [dir for dir in dirs_to_apply_reconverge if not os.path.exists(os.path.join(dir, "vasp.log"))]
        dirs_with_log = [dir for dir in dirs_to_apply_reconverge if os.path.exists(os.path.join(dir, "vasp.log"))]

        # Prioritize directories without vasp.log
        dirs_to_check = dirs_without_log + dirs_with_log
        
        for i, dir in enumerate(dirs_to_check):
            if not check_convergence(dir):
                if i + len(running_queued_job_directories) > self.max_submissions:
                    leftover_calcs_exceeding_queue_limit.append(dir)
                else:
                    self.reconverge(
                        dir, calc_type, HPC, VASP_version, CPU, walltime, cpu_per_node
                    )
                    dirs_to_search_next_time.append(dir)
            else:
                print(f"CONVERGED: {dir}")

        self.update_resubmit_log(
            dirs_to_search_next_time
            + running_queued_job_directories
            + leftover_calcs_exceeding_queue_limit
        )
        return dirs_to_search_next_time

    def load_non_converged_paths(self, from_dataframe_path):
        if from_dataframe_path:
            df = pd.read_pickle(from_dataframe_path)
            return [
                (
                    path.rstrip(os.sep + "OUTCAR")
                    if path.endswith(os.sep + "OUTCAR")
                    else path
                )
                for path in df["filepath"].tolist()
            ]
        return self.reconverge_from_log_file()

    def update_resubmit_log(self, dirs_to_search_next_time):
        with open(os.path.join(self.parent_dir, "resubmit.log"), "w") as log_file:
            for dir_path in dirs_to_search_next_time:
                log_file.write(dir_path + "\n")

    def reconverge(
        self,
        dirpath,
        calc_type="SDRS",
        HPC="Setonix",
        VASP_version="5.4.4",
        CPU=128,
        walltime=24,
        cpu_per_node=128,
    ):
        self.handle_error_run_files(dirpath)
        reconverge_methods = {
            "static": self.reconverge_static,
            "SDRS": self.reconverge_SDRS,
            "DRS": self.reconverge_DRS,
            "base": self.reconverge_base,
        }
        reconverge_method = reconverge_methods.get(calc_type, self.reconverge_base)
        reconverge_method(dirpath, HPC, VASP_version, CPU, walltime, cpu_per_node)

    def handle_error_run_files(self, dirpath):
        error_tar_files_exist = any(
            "error" in f and "tar" in f for f in os.listdir(dirpath)
        )
        if error_tar_files_exist:
            latest_error_run_index = self.find_latest_error_run_index(dirpath)
            error_run_folder_path = os.path.join(
                dirpath, f"error_run_{latest_error_run_index + 1}"
            )
            os.makedirs(error_run_folder_path)
            self.move_files_to_error_run_folder(dirpath, error_run_folder_path)

    def move_files_to_error_run_folder(self, dirpath, error_run_folder_path):
        for f in os.listdir(dirpath):
            if ("error" in f and "tar" in f) or f.endswith(".sh"):
                shutil.move(
                    os.path.join(dirpath, f), os.path.join(error_run_folder_path, f)
                )

        for og_file in ["INCAR.orig", "POSCAR.orig", "KPOINTS.orig", "custodian.json"]:
            if os.path.exists(os.path.join(dirpath, og_file)):
                shutil.move(
                    os.path.join(dirpath, og_file),
                    os.path.join(error_run_folder_path, og_file),
                )

        for current_run in [
            "INCAR",
            "POSCAR",
            "POTCAR",
            "OUTCAR",
            "vasprun.xml",
            "vasp.log",
        ]:
            if os.path.exists(os.path.join(dirpath, current_run)):
                shutil.copy(
                    os.path.join(dirpath, current_run),
                    os.path.join(error_run_folder_path, current_run),
                )

    def find_latest_error_run_index(self, dirpath):
        error_run_indices = [0]
        for f in os.listdir(dirpath):
            if f.startswith("error_run_"):
                try:
                    n = int(f.split("error_run_")[-1])
                    error_run_indices.append(n)
                except ValueError as e:
                    print(f"Exception occurred at {dirpath}: {e}")
        return max(error_run_indices)

    def generate_custodian_string(self, template_filename, user_inputs):
        template_path = os.path.join(self.script_template_dir, template_filename)
        return jobfile._replace_fields(template_path, user_inputs)

    def reconverge_base(self, dirpath, HPC, VASP_version, CPU, walltime, cpu_per_node):
        self.reconverge_generic(
            dirpath, "template_BASE.py", HPC, VASP_version, CPU, walltime, cpu_per_node
        )

    def reconverge_static(
        self, dirpath, HPC, VASP_version, CPU, walltime, cpu_per_node
    ):
        self.reconverge_generic(
            dirpath,
            "template_Static.py",
            HPC,
            VASP_version,
            CPU,
            walltime,
            cpu_per_node,
        )

    def reconverge_DRS(self, dirpath, HPC, VASP_version, CPU, walltime, cpu_per_node):
        stages_left = self.get_stages_left(dirpath, ["relax_1", "relax_2"], 3)
        self.reconverge_generic(
            dirpath,
            "template_DRS.py",
            HPC,
            VASP_version,
            CPU,
            walltime,
            cpu_per_node,
            {"{STAGES_LEFT}": str(stages_left)},
        )

    def reconverge_SDRS(self, dirpath, HPC, VASP_version, CPU, walltime, cpu_per_node):
        stages_left = self.get_stages_left(
            dirpath, ["static_1", "relax_1", "relax_2"], 4
        )
        self.reconverge_generic(
            dirpath,
            "template_SDRS.py",
            HPC,
            VASP_version,
            CPU,
            walltime,
            cpu_per_node,
            {"{STAGES_LEFT}": str(stages_left)},
        )

    def get_stages_left(self, dirpath, stage_markers, default_stages_left):
        for i, marker in enumerate(reversed(stage_markers)):
            if any(f.endswith(f".{marker}") for f in os.listdir(dirpath)):
                return i + 1
        return default_stages_left

    def reconverge_generic(
        self,
        dirpath,
        template_filename,
        HPC,
        VASP_version,
        CPU,
        walltime,
        cpu_per_node,
        extra_inputs=None,
    ):
        user_inputs = {
            "{VASPOUTPUTFILENAME}": '"vasp.log"',
            "{MAXCUSTODIANERRORS}": "20",
        }
        if extra_inputs:
            user_inputs.update(extra_inputs)

        custodian_string = self.generate_custodian_string(
            template_filename, user_inputs
        )
        script_name = os.path.join(
            self.script_template_dir,
            f"{template_filename.split('_')[0]}_Custodian_{HPC}.sh",
        )
        job = jobfile(
            file_path=script_name,
            HPC=HPC,
            VASP_version=VASP_version,
            CPU=CPU,
            walltime=walltime,
            cpu_per_node=cpu_per_node,
            generic_insert_field=["{CUSTODIANSTRING}"],
            generic_insert=[custodian_string],
        )
        target_script_name = f"{os.path.basename(dirpath)}.sh"
        job.to_file(job_name=target_script_name, output_path=dirpath)
        self.submit_to_queue(dirpath, target_script_name)

    def reconverge_from_log_file(self):
        resubmit_log_file = os.path.join(self.parent_dir, "resubmit.log")
        if os.path.isfile(resubmit_log_file):
            with open(resubmit_log_file, "r") as log_file:
                non_converged_dirs = [line.strip() for line in log_file.readlines()]

            largest_n = get_latest_file_iteration(self.parent_dir, "resubmit.log_")
            os.rename(
                resubmit_log_file,
                os.path.join(self.parent_dir, f"resubmit.log_{largest_n + 1}"),
            )

            return non_converged_dirs
        else:
            print("No resubmit log file found. Nothing to resubmit from old logs.")
            return []
