import os
import shutil


def create_folder(directory, delete_folder=False, quiet=True):
    """
    Create a folder if it doesn't exist, and optionally delete it if it does.

    Parameters:
    - directory (str): The path of the folder to be created.
    - delete_folder (bool): If True, delete the folder if it already exists.

    Returns:
    - None
    """
    if os.path.exists(directory):
        if delete_folder:
            if not quiet:
                print("Removing directory...")
            if os.path.isdir(directory):
                shutil.rmtree(directory)
            elif os.path.isfile(directory):
                os.remove(directory)
            else:
                if not quiet:
                    print("Given path is a special file - manually remove")
        else:
            if not quiet:
                # Folder already exists, no replacement/deletion
                print("No replacement/deletion created due to folder existing")
    else:
        os.makedirs(directory)


class jobfile:
    def __init__(
        self,
        file_path,
        HPC="Gadi",
        VASP_version="5.4.4",
        CPU=192,
        cpu_per_node=48,
        RAM=64,
        walltime=999,
        max_resubmissions=999,
        generic_insert_field=[],
        generic_insert=[],
    ):
        """
        Initialize a jobfile instance.

        Parameters:
        - file_path (str): The path to the jobfile template.
        - HPC (str): One of "Gadi", "Setonix", or "Magnus" specifying the high-performance computing system.
        - VASP_version (str): VASP version, defaults to "5.4.4".
        - CPU (int): Number of CPUs to use in the job.
        - cpu_per_node (int): Number of CPUs per node on the HPC system.
            Gadi: 48 is 1 node  (Only use in full nodes, as you are charged for full nodes)
            Magnus: 24 is 1 node (Only use in full nodes, as you are charged for full nodes)
            Setonix: 128 is 1 node (Charged on a per-cpu hour basis, not per-node like Gadi)
        - RAM (int): RAM to be allocated (specified for Gadi only).
        - walltime (int): Walltime of the job in hours.
        - max_resubmissions (int): Maximum number of resubmissions.
        - generic_insert_field (list): List of placeholder fields to be replaced in the jobfile template.
        - generic_insert (list): List of values or file paths corresponding to the placeholder fields.

        Returns:
        - None
        """
        self.file_path = file_path
        self.HPC = HPC
        self.VASP_version = VASP_version
        self.CPU = CPU
        self.RAM = RAM
        self.walltime = walltime
        self.max_resubmissions = max_resubmissions
        self.cpu_per_node = cpu_per_node
        self.generic_insert_field = generic_insert_field
        self.generic_insert = generic_insert

    def to_file(
        self, job_name="template_job", output_path=os.path.join(os.getcwd(), "test")
    ):
        """
        Generate a jobfile by replacing placeholders in the template and insert values from generic_insert.

        Parameters:
        - job_name (str): Name of the generated jobfile.
        - output_path (str): Output path for the generated jobfile.

        Returns:
        - None
        """

        create_folder(output_path)

        with open("%s" % (self.file_path), "r") as fin:
            filedata = fin.read()

        fin = open("%s" % (self.file_path), "rt", newline="\n")

        # Replace the target strings
        replace_dict = {
            "{WALLTIMESTRING}": "%s:00:00" % self.walltime,
            "{CPUSTRING}": str(self.CPU),
            "{MAXCONVITERATIONS}": str(self.max_resubmissions - 1),
            "{MEMORYSTRING}": "%sGB" % self.RAM if self.HPC == "Gadi" else "",
            "{NODESTRING}": (
                "1"
                if self.CPU <= self.cpu_per_node
                else "%s" % int(self.CPU / self.cpu_per_node)
            ),
            "{CASESTRING}": "%s" % job_name,
        }

        for field, value in replace_dict.items():
            filedata = filedata.replace(field, value)

        if self.VASP_version == "5.4.4":
            filedata = filedata.replace(
                "{VASPMODULELOADSTRING}", "module load vasp/%s" % self.VASP_version
            )
        else:
            if self.HPC == "Setonix" and self.VASP_version in ["6.3.0", "6.2.1"]:
                filedata = filedata.replace(
                    "{VASPMODULELOADSTRING}", "module load vasp6/%s" % self.VASP_version
                )
            else:
                filedata = filedata.replace(
                    "{VASPMODULELOADSTRING}", "module load vasp/%s" % self.VASP_version
                )

        # Insert values from generic_insert into corresponding fields
        for insert_field, insert_value in zip(
            self.generic_insert_field, self.generic_insert
        ):
            if os.path.isfile(insert_value):
                # If insert_value is a path, inject the contents of the file
                with open(insert_value, "r") as insert_file:
                    insert_content = insert_file.read()
                filedata = filedata.replace(insert_field, insert_content)
            else:
                # If insert_value is not a path, directly insert the string
                filedata = filedata.replace(insert_field, insert_value)

        # Write the file out again
        with open(os.path.join(output_path, job_name), "w") as fout:
            fout.write(filedata)

        fin.close()
        fout.close()

    @staticmethod
    def _replace_fields(template_path, user_inputs):
        """
        Read a file, replace specified fields with user inputs, and create a jobfile instance.

        Parameters:
        - template_path (str): Path to the template file.
        - user_inputs (dict): Dictionary containing user-specified inputs for field replacements.

        Returns:
        - string containing generated text

        Example:
            template_path = '/cmmc/u/hmai/personal_dev/utils/jobscript_templates/CustodianScripts/SDRS_template.py'
            user_inputs = {
                'VASPOUTPUTFILENAME': 'vasp.log',
                'STAGE': '2',
            }
            created_jobfile_text = _replace_fields(template_path, user_inputs)
        """

        # Read the template file
        with open(template_path, "r") as template_file:
            template_content = template_file.read()

        # Replace specified fields with user inputs
        for field, value in user_inputs.items():
            template_content = template_content.replace(field, str(value))

        return template_content

    def to_string(self):
        """
        Convert the jobfile instance to a string representation.

        Returns:
        - str: String representation of the jobfile content.
        """
        with open(self.file_path, "r") as file:
            content = file.read()

        # Replace placeholders in the content if needed
        # content = content.replace("{SOME_PLACEHOLDER}", str(self.some_attribute))

        return content
