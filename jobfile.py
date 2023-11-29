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
    '''
    Class for jobfile object for passing into createJobFolder

    Attributes:
    file_path: string, is the path to jobfile template (switch for normal VASP calcs vs DDEC6)
    HPC: string, one of "Gadi", "Setonix", or "Magnus"  This ensures that correct jobsubmission script is used.
    VASP_version: string (one of: "5.4.4", "5.4.4-metagga", "6.1.0", "6.2.1", "6.3.0", "6.3.1")
                  NOT ALL ARE AVAILABLE ON ALL SYSTEMS
                  for loading module - defaults to "5.4.4"
                  AVAILABILITY:
                  for Gadi (5.4.4, 5.4.4-metagga, 6.1.0, 6.2.1, 6.3.3)
                  for Magnus (5.4.4, 6.2.0)
                  for Setonix (5.4.4, 6.2.1, 6.3.0)
    CPU: integer
         number of CPUs to use in job,
         Gadi: 48 is 1 node  (Only use in full nodes, as you are charged for full nodes)
         Magnus: 24 is 1 node (Only use in full nodes, as you are charged for full nodes)
         Setonix: 128 is 1 node (Charged on a per-cpu hour basis, not per-node like Gadi)
    RAM: RAM to be allocated - this is only specified in the case of Gadi, Setonix + magnus do not need specification.
    walltime: INTEGER ONLY
              The walltime of the job in hours
    '''
    def __init__(self,
                 file_path,
                 HPC = "Gadi",
                 VASP_version = "5.4.4",
                 CPU = 192,
                 cpu_per_node = 48,
                 RAM = 64,
                 walltime = 999,
                 max_resubmissions = 999):
        self.file_path = file_path
        self.HPC = HPC
        self.VASP_version = VASP_version
        self.CPU = CPU
        self.RAM = RAM
        self.walltime = walltime
        self.max_resubmissions = max_resubmissions
        self.cpu_per_node = cpu_per_node

    def to_file(self,\
                job_name = 'template_job',\
                output_path = os.path.join(os.getcwd(), "test")):
        """
        """

        create_folder(output_path)

        with open("%s" % (self.file_path), 'r') as fin :
            filedata = fin.read()
        fin = open("%s" % (self.file_path), "rt", newline="\n")
        # Replace the target string
        filedata = filedata.replace("{WALLTIMESTRING}", "%s:00:00" % self.walltime)
        filedata = filedata.replace("{CPUSTRING}", str(self.CPU))
        filedata = filedata.replace("{MAXCONVITERATIONS}", str(self.max_resubmissions-1))

        # Only on GADI
        filedata = filedata.replace("{MEMORYSTRING}", "%sGB" % self.RAM)

        if self.CPU <= self.cpu_per_node:
            filedata = filedata.replace("{NODESTRING}", "1")
        else:
            filedata = filedata.replace("{NODESTRING}", "%s" % int(self.CPU/self.cpu_per_node))
            
        filedata = filedata.replace("{CASESTRING}", "%s" % job_name)

        if self.VASP_version == "5.4.4":
            filedata = filedata.replace("{VASPMODULELOADSTRING}", 'module load vasp/%s' %  self.VASP_version)
        else:
            if self.HPC == "Setonix" and self.VASP_version in ["6.3.0", "6.2.1"]:
                filedata = filedata.replace("{VASPMODULELOADSTRING}", 'module load vasp6/%s' % self.VASP_version)
            else:
                filedata = filedata.replace("{VASPMODULELOADSTRING}", 'module load vasp/%s' % self.VASP_version)

        # Write the file out again
        with open(os.path.join(output_path, job_name), 'w') as fout:
            fout.write(filedata)

        fin.close()
        fout.close()