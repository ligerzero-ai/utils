import glob
import os
import shutil
import tarfile
import tempfile

import numpy as np
import pandas as pd

from pymatgen.core import Structure
from pymatgen.io.vasp import Incar, Kpoints, Vasprun

from utils.vasp.parser.outcar import Outcar
# from utils.vasp.vasp import check_convergence
#from utils.vasp.vasp import read_OUTCAR
import warnings
import utils.generic as gen_tools

def check_convergence(directory, filename_vasprun="vasprun.xml", filename_vasplog="vasp.log", backup_vasplog = "error.out"):
    """
    Check the convergence status of a VASP calculation.

    Args:
        directory (str): The directory containing the VASP files.
        filename_vasprun (str, optional): The name of the vasprun.xml file (default: "vasprun.xml").
        filename_vasplog (str, optional): The name of the vasp.log file (default: "vasp.log").

    Returns:
        bool: True if the calculation has converged, False otherwise.

    Raises:
        FileNotFoundError: If neither vasprun.xml nor vasp.log is found.

    Example:
        >>> convergence_status = check_convergence(directory="/path/to/vasp_files")
        >>> if convergence_status:
        ...     print("Calculation has converged.")
        ... else:
        ...     print("Calculation has not converged.")
    """
    try:
        vr = Vasprun(filename=os.path.join(directory, filename_vasprun))
        return vr.converged
    except:
        line_converged = "reached required accuracy - stopping structural energy minimisation"
        try:
            converged = gen_tools.is_line_in_file(filepath=os.path.join(directory, filename_vasplog),
                                        line=line_converged,
                                        exact_match=False)
            return converged
        except:
            try:
                converged = gen_tools.is_line_in_file(filepath=os.path.join(directory, backup_vasplog),
                            line=line_converged,
                            exact_match=False)
                return converged
            except:
                return False
            
def process_error_archives(directory):
    """
    Processes all tar or tar.gz files starting with 'error' in the specified directory and its subdirectories.

    Args:
        directory (str): The directory to search for tar files.
        process_function (function): A user-defined function that operates on the extracted directory.

    Returns:
        None
    """
    error_files = []
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('error') and (file.endswith('.tar') or file.endswith('.tar.gz')):
                error_files.append(os.path.join(root, file))
    
    df_list = []
    for error_file in error_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the tar file into the temporary directory
            with tarfile.open(error_file, "r:*") as tar:
                # print(error_file)
                tar.extractall(path=temp_dir)

            # Apply the user-defined function on the extracted contents
            df = pd.DataFrame(_get_vasp_outputs(temp_dir))
            df_list.append(df)
            # Temporary directory and its contents will be automatically deleted after this block
    
    print(f"Processing error dirs in {directory} complete.")
    if df_list:
        return pd.concat(df_list)
    else:
        return pd.DataFrame()

def _get_vasp_outputs(directory,
                        structure=None,
                        parse_all_in_dir=True):
    # Pattern to find OUTCAR files
    if parse_all_in_dir:
        outcar_files = glob.glob(os.path.join(directory, "OUTCAR*"))
    else:
        outcar_files = glob.glob(os.path.join(directory, "OUTCAR"))
    data = []
    for outcar_file in outcar_files:
        suffix = os.path.basename(outcar_file).replace("OUTCAR", "")
        if structure is None:
            structure = get_structure(directory)
            
        # Initialize dictionary to hold file data
        file_data = {
            "POSCAR": structure,
            "OUTCAR": np.nan,
            "INCAR": np.nan,
            "KPOINTS": np.nan
        }
        
        if os.path.isfile(outcar_file):
            try:
                outcar = Outcar()
                outcar.from_file(filename = outcar_file)
                file_data["OUTCAR"] = outcar
            except Exception as e:
                print(f"Error reading OUTCAR file {outcar_file}: {e}")
                
        # Try to find INCAR file with same suffix
        incar_file = os.path.join(directory, f"INCAR{suffix}")
        if os.path.isfile(incar_file):
            try:
                incar = Incar.from_file(incar_file).as_dict()
                file_data["INCAR"] = incar
            except Exception as e:
                print(f"Error reading INCAR file {incar_file}: {e}")

        # Try to find KPOINTS file with same suffix
        kpoints_file = os.path.join(directory, f"KPOINTS{suffix}")
        if os.path.isfile(kpoints_file):
            try:
                kpoints = Kpoints.from_file(kpoints_file).as_dict()
                file_data["KPOINTS"] = kpoints
            except Exception as e:
                print(f"Error reading KPOINTS file {kpoints_file}: {e}")
                
        data.append(file_data)
    return pd.DataFrame(data)

def get_SCF_cycle_convergence(outcar_scf_arrays, threshold=1e-5):
    diff = outcar_scf_arrays[-1] - outcar_scf_arrays[-2]
    if abs(diff) < threshold:
        return True
    else:
        return False
    
def _get_KPOINTS_info(KPOINTS, INCAR):
    if np.isnan(KPOINTS):
        kpoints_key = 'KSPACING'
        if kpoints_key in INCAR:
            kpoints = f"KSPACING: {INCAR[kpoints_key]}"
        else:
            kpoints = "KSPACING: 0.5"
    else:
        kpoints = KPOINTS
    return kpoints
    
def process_outcar(outcar, structure):
    try:
        energies = outcar.parse_dict["energies"]
    except:
        energies = np.nan
        
    try:
        ionic_step_structures = np.array([Structure(cell, structure.species, outcar.parse_dict["positions"][i], coords_are_cartesian=True).to_json()
                                            for i, cell in enumerate(outcar.parse_dict["cells"])])
    except:
        ionic_step_structures = np.nan
    
    try:
        energies_zero =  outcar.parse_dict["energies_zero"]
    except:
        energies_zero = np.nan
        
    try:
        forces = outcar.parse_dict["forces"]
    except:
        forces = np.nan
        
    try:
        stresses = outcar.parse_dict["stresses"]
    except:
        stresses = np.nan
        
    try:
        magmoms = np.array(outcar.parse_dict["final_magmoms"])
    except:
        magmoms = np.nan
        
    try:
        scf_steps = [len(i) for i in outcar.parse_dict["scf_energies"]]
        scf_conv_list = [get_SCF_cycle_convergence(d, threshold=1e-5) for d in outcar.parse_dict["scf_energies"]]
    except:
        scf_steps = np.nan
        scf_conv_list = np.nan
    
    try:
        calc_start_time = outcar.parse_dict["execution_datetime"]
    except:
        calc_start_time = np.nan
    
    try:
        consumed_time = outcar.parse_dict["resources"]
    except:
        consumed_time = np.nan
        
    df = pd.DataFrame([[calc_start_time,
                        consumed_time,
                        ionic_step_structures,
                        energies,
                        energies_zero,
                        forces,
                        stresses,
                        magmoms,
                        scf_steps,
                        scf_conv_list]],
                columns = ["calc_start_time",
                           "consumed_time",
                           "structures",
                            "energy",
                            "energy_zero",
                            "forces",
                            "stresses",
                            "magmoms",
                            "scf_steps",
                            "scf_convergence"])    
    return df

def get_structure(directory):
    """
    Attempts to read the structure from various file names in the specified order.

    Args:
        directory (str): The directory where the files are located.

    Returns:
        pymatgen.core.Structure: The structure object if successful, None otherwise.
    """
    structure_filenames = [
        "CONTCAR",
        "POSCAR",
    ] + glob.glob(os.path.join(directory, "starter*.vasp"))

    for filename in structure_filenames:
        try:
            return Structure.from_file(os.path.join(directory, filename))
        except Exception as e:
            print(f"Failed to parse structure file {filename}: {e}")

    print("Failed to parse appropriate structure file completely")
    return np.nan

def get_vasp_outputs(directory,
                     extract_error_dirs=True,
                     parse_all_in_dir=True):
    
    df_direct_outputs = _get_vasp_outputs(directory,
                                          parse_all_in_dir=parse_all_in_dir)
    if extract_error_dirs:
        df_error_outputs = process_error_archives(directory)
    else:
        df_error_outputs = pd.DataFrame()
    df_all = pd.concat([df_direct_outputs, df_error_outputs])
    return df_all

def grab_electron_info(directory_path, line_before_elec_str="PAW_PBE", potcar_filename = "POTCAR"):
    
    structure = get_structure(directory_path)
    if structure != None:
        element_list, element_count = element_count_ordered(structure)
        
    electron_of_potcar = []
    
    with open(os.path.join(directory_path, potcar_filename), 'r') as file:
        lines = file.readlines()  # Read the lines from the file
        should_append = False  # Flag to determine if the next line should be appended
        for line in lines:
            stripped_line = line.strip()  # Remove leading and trailing whitespace
            if should_append:
                electron_of_potcar.append(float(stripped_line))
                should_append = False  # Reset the flag
            if stripped_line.startswith(line_before_elec_str):
                should_append = True  # Set the flag to append the next line
        
    return element_list, element_count, electron_of_potcar

def get_total_electron_count(directory_path, line_before_elec_str="PAW_PBE", potcar_filename = "POTCAR"):
    ele_list, ele_count, electron_of_potcar = grab_electron_info(directory_path=directory_path, line_before_elec_str=line_before_elec_str, potcar_filename=potcar_filename)
    total_electron_count = np.dot(ele_count, electron_of_potcar)
    return total_electron_count

def element_count_ordered(structure):
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


def parse_vasp_directory(directory,
                         extract_error_dirs=True,
                         parse_all_in_dir=True):
    df = get_vasp_outputs(directory,
                          extract_error_dirs=extract_error_dirs,
                          parse_all_in_dir=parse_all_in_dir)
    df = df.dropna(subset=["OUTCAR"])
    results_df = []
    kpoints_list = []
    for _, row in df.iterrows():
        results_df.append(process_outcar(row.OUTCAR, row.POSCAR))
        kpoints_list.append(_get_KPOINTS_info(row.KPOINTS,row.INCAR))
    try:
        results_df = pd.concat(results_df).sort_values(by="calc_start_time")
    except Exception as e:
        warnings.warn(f"WARNING: {directory} OUTCAR parsing failed!\nFailed with exception {e}")
        results_df = pd.DataFrame()
    results_df["KPOINTS"] = kpoints_list
    results_df = results_df.copy().reset_index(drop=True)
    results_df["INCAR"] = df["INCAR"].tolist()
    
    try:
        element_list, element_count, electron_of_potcar = grab_electron_info(directory_path=directory,
                                                                            potcar_filename="POTCAR")
    except:
        element_list = np.nan
        element_count = np.nan
        electron_of_potcar = np.nan

    try:
        electron_count = get_total_electron_count(directory_path=directory)
    except Exception as e:
        print(e)
        electron_count = np.nan
        
    results_df["element_list"] = [element_list] * len(results_df)
    results_df["element_count"] = [element_count] * len(results_df)
    results_df["potcar_electron_count"] = [electron_of_potcar] * len(results_df)
    results_df["job_name"] = [os.path.basename(directory)] * len(results_df)
    results_df["filepath"] = [directory] * len(results_df)
    results_df["convergence"] = [check_convergence(directory)] * len(results_df)
    return results_df