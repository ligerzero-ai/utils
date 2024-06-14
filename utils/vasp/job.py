import os
import shutil
import io

potcar_library_path = "/root/POTCAR_Library/GGA"
potcar_library_path = "/cmmc/u/hmai/pyiron-resources-cmmc/vasp/potentials/potpaw_PBE"


def createFolder(directory, delete_folder="no"):
    import os
    import shutil

    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        if delete_folder == "no":
            # print('no replacement/deletion created due to folder existing')
            x = 1
        else:
            print("removing directory...")
            if os.path.isdir(directory):
                shutil.rmtree(directory)
            elif os.path.isfile(directory):
                os.rm(directory)
            else:
                print("given path is a special file - manually remove")


def get_immediate_subdirectories(a_dir):
    return [
        f.path
        for f in os.scandir(a_dir)
        if f.is_dir() and os.path.basename(f) != ".ipynb_checkpoints"
    ]


class jobfile:
    """
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
                  for Garching (5.4.4, 6.4.0)
    CPU: integer
         number of CPUs to use in job,
         Gadi: 48 is 1 node  (Only use in full nodes, as you are charged for full nodes)
         Magnus: 24 is 1 node (Only use in full nodes, as you are charged for full nodes)
         Setonix: 128 is 1 node (Charged on a per-cpu hour basis, not per-node like Gadi)
    RAM: RAM to be allocated - this is only specified in the case of Gadi, Setonix + magnus do not need specification.
    walltime: INTEGER ONLY
              The walltime of the job in hours
    """

    def __init__(
        self,
        file_path,
        HPC="Gadi",
        VASP_version="5.4.4",
        CPU=192,
        RAM=64,
        walltime=999,
        max_resubmissions=999,
    ):
        self.file_path = file_path
        self.HPC = HPC
        self.VASP_version = VASP_version
        self.CPU = CPU
        self.RAM = RAM
        self.walltime = walltime
        self.max_resubmissions = max_resubmissions

    def to_file(
        self, case_name="template_job", output_path=os.path.join(os.getcwd(), "test")
    ):
        """
        Writes KPOINTS file with MP gamma centred grid:

        case_name = string at top of file (defaults to "no filename given")
        filepath = system filepath where KPOINTS is to be written

        """

        createFolder(output_path)

        with open("%s" % (self.file_path), "r") as fin:
            filedata = fin.read()
        if self.HPC == "Gadi":
            fin = open("%s" % (self.file_path), "rt", newline="\n")
        elif self.HPC in ["Setonix", "Magnus"]:
            fin = open("%s" % (self.file_path), "rt", newline="\n")
        # Replace the target string
        filedata = filedata.replace("{WALLTIMESTRING}", "%s:00:00" % self.walltime)
        filedata = filedata.replace("{CPUSTRING}", str(self.CPU))
        filedata = filedata.replace(
            "{MAXCONVITERATIONS}", str(self.max_resubmissions - 1)
        )

        # Only on GADI
        filedata = filedata.replace("{MEMORYSTRING}", "%sGB" % self.RAM)

        # Only on MAGNUS/SETONIX
        if self.HPC == "Magnus":
            max_cpu_count = 24
        elif self.HPC == "Setonix":
            max_cpu_count = 128
        elif self.HPC == "Garching":
            max_cpu_count = 40
        if self.CPU <= max_cpu_count:
            filedata = filedata.replace("{NODESTRING}", "1")
        else:
            filedata = filedata.replace(
                "{NODESTRING}", "%s" % int(self.CPU / max_cpu_count)
            )

        filedata = filedata.replace("{CASESTRING}", "%s" % case_name)

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
            if self.HPC == "Garching":
                # vasp/5.3-constrainedcollinearmagnetism  vasp/5.4.4-buildFeb20  vasp/5.4.4-elphon        vasp/5.4.4-python  vasp/6.4.0-buildMar23
                # vasp/5.4.4      vasp/5.4.4-Dudarev     vasp/5.4.4-potentiostat  vasp/6.4.0         vasp/6.4.0-python
                filedata = filedata.replace(
                    "{VASPMODULELOADSTRING}", "module load vasp/%s" % self.VASP_version
                )

        # Write the file out again
        with open(os.path.join(output_path, case_name), "w") as fout:
            fout.write(filedata)

        fin.close()
        fout.close()


def stackElementString(structure):
    site_element_list = [site.species_string for site in structure]
    past_element = site_element_list[0]
    element_list = [past_element]
    element_count = []
    count = 0
    for element in site_element_list:
        if element == past_element:
            count += 1
        else:
            element_count.append(count)
            element_list.append(element)
            count = 1
            past_element = element
    element_count.append(count)
    return element_list, element_count


def createPOTCAR(structure, path=os.getcwd()):

    element_list = stackElementString(structure)[0]
    potcar_paths = []

    for element in element_list:
        if element == "Nb":
            element = "Nb_sv"  # Use 13 electron
            element = "Nb_pv"  # Use 11 electron
        elif element == "K":
            element = "K_sv"  # 9 electron
            element = "K_pv"  # 7 electron
        elif element == "Ca":
            element = "Ca_sv"  # 9 electron
            element = "Ca_pv"  # 7 electron
        elif element == "Rb":
            element = "Rb_sv"  # 9 electron
            element = "Rb_pv"  # 7 electron
        elif element == "Sr":
            element = "Sr_sv"  # 9 electron
        elif element == "Cs":
            element = "Cs_sv"  # 9 electron
        elif element == "Ba":
            element = "Ba_sv"  # 10 electron
        elif element == "Fr":
            element = "Fr_sv"  # 9 electron
        elif element == "Ra":
            element = "Ra_sv"  # 9 electron
        elif element == "Y":
            element = "Y_sv"  # 9 electron
        elif element == "Zr":
            element = "Zr_sv"  # 10 electron
        elif element == "Fr":
            element = "Fr_sv"  # 9 electron
        elif element == "Ra":
            element = "Ra_sv"  # 9 electron
        elif element == "Y":
            element = "Y_sv"  # 9 electron

        potcar_paths.append(os.path.join(potcar_library_path, element, "POTCAR"))

    with open(os.path.join(path, "POTCAR"), "wb") as wfd:
        for f in potcar_paths:
            with open(f, "rb") as fd:
                shutil.copyfileobj(fd, wfd)


class KPOINTS:
    """
    Class for KPOINTS object for passing into createJobFolder

    Attributes:

    name: String for naming the first line of KPOINTS, purely decorative
    subdivs: Assign the subdivision splits along reciprocal lattice, input as list e.g. [4, 2, 1]
    shift: optional shift of mesh, input as list e.g. [0, 0, 0]

    """

    def __init__(self, subdivs, shift):
        self.subdivs = subdivs
        self.shift = shift

    def to_file(self, case_name="KPOINTS", filepath=os.getcwd()):
        """
        Writes KPOINTS file with MP gamma centred grid:

        case_name = string at top of file (defaults to "no filename given")
        filepath = system filepath where KPOINTS is to be written

        """
        createFolder(filepath)
        f = io.open(os.path.join(filepath, "KPOINTS"), "w", newline="\n")
        with open(os.path.join(filepath, "KPOINTS"), "a", newline="\n") as f:
            # File name (just string on first line of KPOINTS)
            f.write("%s\n" % case_name)
            # Use automatic generation "0"
            f.write("0\n")
            # Monkhorst-Pack Gamma centred grid
            f.write("Gamma\n")
            # Subdivisions along reciprocal lattice vectors
            subdiv_string = ""
            for i in self.subdivs:
                subdiv_string += "%s " % str(i)
            f.write("%s\n" % subdiv_string)
            # optional shift of the mesh (s_1, s_2, s_3)
            shift_string = ""
            for i in self.shift:
                shift_string += "%s " % str(i)
            f.write("%s\n" % shift_string)
        f.close()


def createJobFolder(
    structure,
    KPOINT=None,
    folder_path=os.path.join(os.getcwd(), "jobfolder"),
    INCAR=None,
    jobfile=None,
    quiet=True,
):
    # This assumes that incar file base is present already, please adjust this function to adjust the incar flags
    # creates a subdirectory of chosen name in current directory
    parent_folder = os.getcwd()
    createFolder(folder_path)

    structure.to(
        fmt="poscar",
        filename=os.path.join(
            folder_path, f"starter-{os.path.basename(folder_path)}.vasp"
        ),
    )
    structure.to(fmt="poscar", filename=os.path.join(folder_path, "POSCAR"))

    createPOTCAR(structure, path="%s" % folder_path)

    INCAR.write_file(os.path.join(folder_path, "INCAR"))

    if KPOINT:
        KPOINT.to_file(filepath=folder_path)

    jobfile.to_file(
        case_name="%s.sh" % os.path.basename(folder_path),
        output_path="%s" % (folder_path),
    )
    if not quiet:
        print("Generating jobfolder, name %s" % (os.path.basename(folder_path)))
