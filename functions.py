import os
import shutil
import io

from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Potcar, Incar, Kpoints

from utils.jobfile import jobfile
potcar_library_path = "/root/POTCAR_Library/GGA"
potcar_library_path = "/u/hmai/pyiron-resources-cmmc/vasp/potentials/potpaw_PBE"

sites_to_study = {"S11-RA110-S3-32": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                  "S3-RA110-S1-11": [20, 22, 24, 26, 28, 30, 32, 34, 36],
                  "S3-RA110-S1-12": [12, 14, 16, 18, 20, 22, 24],
                  "S5-RA001-S210": [24, 27, 29, 31, 33, 35, 37],
                  "S5-RA001-S310": [23, 27, 33, 37, 40],
                  "S9-RA110-S2-21": list(range(23, 37))}

def structures_from_vasp_folder(folder_path):
    # Initialize an empty dictionary to store the structures
    structures_dict = {}

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .vasp file
        if filename.endswith('.vasp'):
            # Read in the Structure from the VASP file using pymatgen's Structure class
            structure = Structure.from_file(os.path.join(folder_path, filename))
            # Strip the .vasp extension from the filename and use it as the dictionary key
            key = os.path.splitext(filename)[0]
            # Assign the Structure object to the dictionary with the key
            structures_dict[key] = structure
    
    # Return the dictionary containing the structures
    return structures_dict

def createFolder(directory, delete_folder='no'):
    import os; import shutil
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        if delete_folder == 'no':
            #print('no replacement/deletion created due to folder existing')
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
    return [f.path for f in os.scandir(a_dir) if f.is_dir() and os.path.basename(f) != ".ipynb_checkpoints"]

def generateINCAR(structure,
                  path = os.path.join(os.getcwd(), "INCAR"),
                  ISIF = 2,
                  ISPIN = 1,
                  ENCUT = 350,
                  EDIFF = 1E-4,
                  EDIFFG = -0.02,
                  NCORE = 4,
                  KPAR = 1,
                  SYSTEM = "filltext",
                  functional = 'PBE',
                  reverse_magmom = False,
                  base_element = "Fe"):

    INCAR_file = Incar()
    INCAR_file = INCAR_file.from_file(path)
    INCAR_file["SYSTEM"] = SYSTEM
    INCAR_file["ISIF"] = ISIF
    INCAR_file["ENCUT"] = ENCUT
    INCAR_file["EDIFF"] = EDIFF
    INCAR_file["EDIFFG"] = EDIFFG
    INCAR_file["NCORE"] = NCORE
    INCAR_file["KPAR"] = KPAR

    dictionary_of_functionals = {"PW91" : '91',
                                "PBE" : 'PE',
                                "AM05" : 'AM',
                                "PBEsol": 'PS',
                                "Hendin-Lundquist" : "HL",
                                "Ceperley-Alder" : "CA",
                                "Perdew-Zunger" : "PZ",
                                "Wigner" : 'WI',
                                "Revised-PBE-Pade" : "RP",
                                "revPBE" : "RE",
                                "Vosko-Wilk-Nusair" : "VW",
                                "B3LYP-LDA-VWN3" : "B3",
                                "B3LYP-LDA-BWN5" : "B5",
                                "BEEF" : "BF",
                                "no-xc" : "CO"}

    # These magmoms are from projects past and present... Feel free to alter them
    # Ni-H from Ni-GB manuscript
    #     dictionary_of_magmom = {"Ni" : 2.0,
    #                             "H"  : 0.0}
    # rest from Fe-bulk manuscript
    dictionary_of_magmom = {'Ac': -0.196,
                         'Ag': 0.114,
                         'Al': -0.17,
                         'Ar': 0.354,
                         'As': -0.136,
                         'At': -0.084,
                         'Au': 0.308,
                         'Ba': -0.25,
                         'Bi': -0.302,
                         'Br': 0.158,
                         'Ca': -0.494,
                         'Cd': -0.158,
                         'Ce': -0.928,
                         'Cl': 0.286,
                         'Co': 3.37,
                         'Cr': -3.71,
                         'Cs': 0.06,
                         'Cu': 0.238,
                         'Dy': 9.11,
                         'Er': 5.048,
                         'Eu': -13.498,
                         'Fe': 3.0,
                         'Fr': -0.046,
                         'Ga': -0.4,
                         'Gd': -14.248,
                         'Ge': -0.258,
                         'Hf': -1.17,
                         'Hg': -0.1,
                         'Ho': 6.942,
                         'I': -0.024,
                         'In': -0.51,
                         'Ir': 0.756,
                         'K': 0.152,
                         'Kr': 0.384,
                         'La': -0.416,
                         'Lu': -0.544,
                         'Mg': -0.128,
                         'Mn': -4.128,
                         'Mo': -1.662,
                         'Na': -0.09,
                         'Nb': -1.518,
                         'Nd': -6.142,
                         'Ne': 0.02,
                         #'Ne': -3.0,
                         'Ni': 1.774,
                         'Os': -0.224,
                         'P': -0.112,
                         'Pa': -1.184,
                         'Pb': -0.41,
                         'Pd': 0.73,
                         'Pm': -8.76,
                         'Po': -0.188,
                         'Pr': -3.256,
                         'Pt': 0.74,
                         'Ra': -0.096,
                         'Rb': 0.11,
                         'Re': -1.27,
                         'Rh': 1.194,
                         'Rn': 0.032,
                         'Ru': 0.454,
                         'S': 0.082,
                         'Sb': -0.186,
                         'Sc': -1.12,
                         'Se': -0.008,
                         'Si': -0.194,
                         'Sm': -10.964,
                         'Sn': -0.426,
                         'Sr': -0.128,
                         'Ta': -1.588,
                         'Tb': -12.568,
                         'Tc': -1.208,
                         'Te': -0.13,
                         'Th': -0.508,
                         'Ti': -1.93,
                         'Tl': -0.45,
                         'Tm': 2.776,
                         'U': -2.76,
                         'V': -2.86,
                         'W': -1.606,
                         'Xe': 0.288,
                         'Y': -0.668,
                         'Yb': 0.414,
                         'Zn': -0.196,
                         'Zr': -0.888,
                         'H' : -0.018,
                         'He': -0.010,
                         'Li': -0.168,
                         'Be': -0.302,
                         'B' : -0.314,
                         'C' : -0.204,
                         'N' : 0.094,
                         'O' : 0.454,
                         'F' : 0.348}
    ele_list, ele_count = stackElementString(structure)

    # This is a funny quirk involving 4d metals - we have to adjust the LMAXMIX flag for faster convergence
    if [i for i in ["Mo", "Nb"] if i in ele_list]:
        #print("Mo/Nb present, LMAXMIX = 4 adjustment")
        INCAR_file["LMAXMIX"] = 4
    elif "W" in ele_list:
        #print("W present, LMAXMIX = 6 adjustment")
        INCAR_file["LMAXMIX"] = 6
    else:
        INCAR_file.pop('LMAXMIX',None)

    if ISPIN != 2:
        INCAR_file.pop("MAGMOM",None)
        INCAR_file.pop('BMIX_MAG',None)
        INCAR_file.pop('AMIX_MAG',None)
    else:
        incar_magmom_str = ''
        for idx, element in enumerate(ele_list):
            if reverse_magmom:
                if element == base_element:
                    incar_magmom_str += "%s*%s " % (ele_count[idx], dictionary_of_magmom[ele_list[idx]])
                else:
                    incar_magmom_str += "%s*%s " % (ele_count[idx], -dictionary_of_magmom[ele_list[idx]])
            else:
                incar_magmom_str += "%s*%s " % (ele_count[idx], dictionary_of_magmom[ele_list[idx]])
        INCAR_file["MAGMOM"] = incar_magmom_str


    if functional == "LDA":
        INCAR_file.pop('GGA', None)
    else:
        INCAR_file["GGA"] = dictionary_of_functionals[functional]
        # print('functional key is %s' % dictionary_of_functionals[functional])

    return INCAR_file

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

def createPOTCAR(structure, path = os.getcwd()):

    element_list = stackElementString(structure)[0]
    potcar_paths = []

    for element in element_list:
        if element == "Nb":
            element = "Nb_sv" # Use 13 electron
            element = "Nb_pv" # Use 11 electron
        elif element == "K":
            element = "K_sv" # 9 electron
            element = "K_pv" # 7 electron
        elif element == "Ca":
            element = "Ca_sv" # 9 electron
            element = "Ca_pv" # 7 electron
        elif element == "Rb":
            element = "Rb_sv" # 9 electron
            element = "Rb_pv" # 7 electron
        elif element == "Sr":
            element = "Sr_sv" # 9 electron
        elif element == "Cs":
            element = "Cs_sv" # 9 electron
        elif element == "Ba":
            element = "Ba_sv" # 10 electron
        elif element == "Fr":
            element = "Fr_sv" # 9 electron
        elif element == "Ra":
            element = "Ra_sv" # 9 electron
        elif element == "Y":
            element = "Y_sv" # 9 electron
        elif element == "Zr":
            element = "Zr_sv" # 10 electron
        elif element == "Fr":
            element = "Fr_sv" # 9 electron
        elif element == "Ra":
            element = "Ra_sv" # 9 electron
        elif element == "Y":
            element = "Y_sv" # 9 electron

        potcar_paths.append(os.path.join(potcar_library_path, element, "POTCAR"))

    with open(os.path.join(path, "POTCAR"),'wb') as wfd:
        for f in potcar_paths:
            with open(f,'rb') as fd:
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

    def to_file(self,\
                case_name = 'KPOINTS',\
                filepath = os.getcwd()):
        """
        Writes KPOINTS file with MP gamma centred grid:

        case_name = string at top of file (defaults to "no filename given")
        filepath = system filepath where KPOINTS is to be written

        """
        createFolder(filepath)
        f = io.open(os.path.join(filepath, "KPOINTS"), 'w', newline='\n')
        with open(os.path.join(filepath, "KPOINTS"), 'a', newline='\n') as f:
            # File name (just string on first line of KPOINTS)
            f.write('%s\n' % case_name)
            # Use automatic generation "0"
            f.write('0\n')
            # Monkhorst-Pack Gamma centred grid
            f.write('Gamma\n')
            # Subdivisions along reciprocal lattice vectors
            subdiv_string = ''
            for i in self.subdivs:
                subdiv_string += "%s " % str(i)
            f.write('%s\n' % subdiv_string)
            # optional shift of the mesh (s_1, s_2, s_3)
            shift_string = ''
            for i in self.shift:
                shift_string += "%s " % str(i)
            f.write('%s\n' % shift_string)
        f.close()

def createJobFolder(structure,\
                    KPOINT = None,\
                    folder_path = os.path.join(os.getcwd(), "jobfolder"),\
                    INCAR = None,\
                    jobfile = None,\
                    quiet=True):
    # This assumes that incar file base is present already, please adjust this function to adjust the incar flags
    # creates a subdirectory of chosen name in current directory
    parent_folder = os.getcwd()
    createFolder(folder_path)

    structure.to(fmt="poscar", filename = os.path.join(folder_path, f"starter-{os.path.basename(folder_path)}.vasp"))
    structure.to(fmt="poscar", filename = os.path.join(folder_path, "POSCAR"))

    createPOTCAR(structure, path = "%s" % folder_path)

    INCAR.write_file(os.path.join(folder_path, "INCAR"))

    if KPOINT:
        KPOINT.to_file(filepath = folder_path)

    jobfile.to_file(job_name = '%s.sh' % os.path.basename(folder_path),\
                    output_path = "%s" % (folder_path))
    if not quiet:
        print("Generating jobfolder, name %s" % (os.path.basename(folder_path)))
