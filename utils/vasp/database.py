import os
import glob
import time

from pymatgen.core import Structure
from pymatgen.io.vasp import Vasprun, Kpoints, Incar

import numpy as np
import pandas as pd

import utils.generic as gen_tools
from utils.parallel import parallelise
from utils.vasp.parser.outcar import Outcar
from utils.vasp.parser.output import parse_vasp_directory


def find_vasp_directories(
    parent_dir,
    filenames=[
        "vasp.log",
        "INCAR",
        "POTCAR",
        "CONTCAR",
        "KPOINTS",
        "OUTCAR",
        "vasprun.xml",
    ],
    all_present=False,
    extract_tarballs=True,
    tarball_extensions=(".tar.gz"),
):
    """
    Finds directories in a parent directory that contain specified files.

    Parameters:
        parent_dir (str): The path of the parent directory to search for directories.
        filenames (list): A list of filenames to search for within the directories.
        all_present (bool, optional): Determines whether all the specified filenames should be present in each directory to be considered a match. Defaults to True.

    Returns:
        list: A list of directories that contain the specified files.

    Usage:
        # Find directories containing specific files, requiring all files to be present
        find_directories_with_files("/path/to/parent", ["file1.txt", "file2.jpg"])

        # Find directories containing specific files, requiring at least one file to be present
        find_directories_with_files("/path/to/parent", ["file1.txt", "file2.jpg"], all_present=False)

    Note:
        - The function searches for directories in the `parent_dir` directory using the `os.walk` function.
        - It checks if the specified `filenames` are present in each directory.
        - If `all_present` is True, the function includes only the directories that contain all the specified files.
        - If `all_present` is False, the function includes the directories that contain at least one of the specified files.
        - The function returns a list of directories that meet the specified conditions.
    """
    if extract_tarballs:
        gen_tools.find_and_extract_files_from_tarballs_parallel(
            parent_dir=parent_dir,
            extension=tarball_extensions,
            filenames=filenames,
            suffix=None,
            prefix=None,
        )

    directories = gen_tools.find_directories_with_files(
        parent_dir=parent_dir, filenames=filenames, all_present=all_present
    )

    return directories


def read_OUTCAR(filename="OUTCAR", free_energy=True, energy_zero=True, structures=True):
    """
    Read information from the OUTCAR file and related VASP structure files.

    Parameters:
        filename (str, optional): The path of the OUTCAR file to read. Defaults to "OUTCAR".

    Returns:
        pandas.DataFrame: A DataFrame containing the parsed data from the OUTCAR and related structure files.

    Usage:
        # Read data from the default OUTCAR file "OUTCAR"
        df = read_OUTCAR()

        # Read data from a specific OUTCAR file
        df = read_OUTCAR("/path/to/OUTCAR")

    Note:
        - The function attempts to read information from the specified OUTCAR file using the `Outcar` class from pymatgen.
        - If successful, it collects data such as energies, ionic step structures, forces, stresses, magnetization moments, and SCF step counts.
        - The function searches for related structure files (with extensions .vasp, CONTCAR, and POSCAR) in the same directory as the OUTCAR file.
        - If a related structure file is found, it is parsed using the `Structure` class from pymatgen.
        - The parsed data is stored in a pandas DataFrame with columns for job name, file path, ionic step structures, energies, forces, stresses, magnetization moments, SCF step counts, and convergence.
        - If any part of the parsing encounters an error, the corresponding DataFrame entry will have NaN values.
    """
    outcar = Outcar()
    outcar.from_file(filename=filename)

    structure_name = os.path.basename(os.path.dirname(filename))

    try:
        energies = outcar.parse_dict["energies"]
    except:
        energies = np.nan

    # create a list of file extensions to search for
    extensions = [".vasp", "CONTCAR", "POSCAR"]
    # create an empty list to store matching files
    structure_files = []
    # walk through the directory and check each file's name
    for root, dirs, files in os.walk(os.path.dirname(filename)):
        for file in files:
            if any(extension in file for extension in extensions):
                structure_files.append(os.path.join(root, file))
    for structure_file in structure_files:
        try:
            structure = Structure.from_file(structure_file)
            break
        except:
            pass

    try:
        ionic_step_structures = np.array(
            [
                Structure(
                    cell,
                    structure.species,
                    outcar.parse_dict["positions"][i],
                    coords_are_cartesian=True,
                ).to_json()
                for i, cell in enumerate(outcar.parse_dict["cells"])
            ]
        )
    except:
        ionic_step_structures = np.nan

    try:
        energies_zero = outcar.parse_dict["energies_zero"]
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
    except:
        scf_steps = np.nan

    df = pd.DataFrame(
        [
            [
                structure_name,
                filename,
                ionic_step_structures,
                energies,
                energies_zero,
                forces,
                stresses,
                magmoms,
                scf_steps,
            ]
        ],
        columns=[
            "job_name",
            "filepath",
            "structures",
            "energy",
            "energy_zero",
            "forces",
            "stresses",
            "magmoms",
            "scf_steps",
        ],
    )
    return df


def parse_VASP_directory(
    directory,
    INCAR_filename="INCAR",
    KPOINTS_filename="KPOINTS",
    POTCAR_filename="POTCAR",
    OUTCAR_filename="OUTCAR",
    vasprunxml_filename="vasprun.xml",
    vasplog_filename="vasp.log",
):

    # Find file matching pattern
    structure_files = glob.glob(os.path.join(directory, "starter*.vasp"))

    if len(structure_files) > 0:
        init_structure = Structure.from_file(structure_files[0])
    else:
        init_structure = None

    # OUTCAR first
    try:
        df = read_OUTCAR(filename=os.path.join(directory, OUTCAR_filename))
    except:
        df = pd.DataFrame(
            [
                [
                    os.path.basename(directory),
                    directory,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            ],
            columns=[
                "job_name",
                "filepath",
                "structures",
                "energy",
                "energy_zero",
                "forces",
                "stresses",
                "magmoms",
                "scf_steps",
            ],
        )

    convergence = check_convergence(
        directory=directory,
        filename_vasprun=vasprunxml_filename,
        filename_vasplog=vasplog_filename,
    )
    # INCAR
    try:
        incar = Incar.from_file(os.path.join(directory, INCAR_filename)).as_dict()
    except:
        incar = np.nan

    try:
        # KPOINTS
        kpoints = Kpoints.from_file(os.path.join(directory, KPOINTS_filename)).as_dict()
    except:
        try:
            kspacing = incar["KSPACING"]
            kpoints = f"KSPACING: {kspacing}"
        except:
            kpoints = np.nan

    try:
        element_list, element_count, electron_of_potcar = grab_electron_info(
            directory_path=directory, potcar_filename=POTCAR_filename
        )
    except:
        element_list = np.nan
        element_count = np.nan
        electron_of_potcar = np.nan

    try:
        electron_count = get_total_electron_count(directory_path=directory)
    except:
        electron_count = np.nan

    df["element_list"] = [element_list]
    df["element_count"] = [element_count]
    df["potcar_electron_count"] = [electron_of_potcar]
    df["total_electron_count"] = [electron_count]
    df["convergence"] = [convergence]

    df["kpoints"] = [kpoints]
    df["incar"] = [incar]

    return df


def check_convergence(
    directory,
    filename_vasprun="vasprun.xml",
    filename_vasplog="vasp.log",
    backup_vasplog="error.out",
):
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
        line_converged = (
            "reached required accuracy - stopping structural energy minimisation"
        )
        try:
            converged = gen_tools.is_line_in_file(
                filepath=os.path.join(directory, filename_vasplog),
                line=line_converged,
                exact_match=False,
            )
            return converged
        except:
            try:
                converged = gen_tools.is_line_in_file(
                    filepath=os.path.join(directory, backup_vasplog),
                    line=line_converged,
                    exact_match=False,
                )
                return converged
            except:
                return False


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


def _try_read_structure(
    directory_path, structure_filenames=["CONTCAR", ".vasp", "POSCAR"]
):
    structure_files = []
    # walk through the directory and check each file's name
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if any(file.endswith(filename) for filename in structure_filenames):
                structure_files.append(os.path.join(root, file))
    structure = None
    for structure_file in structure_files:
        try:
            structure = Structure.from_file(structure_file)
            break
        except:
            pass
        if structure == None:
            print(f"no structure found in {directory_path}")
            structure = np.nan
    return structure


def grab_electron_info(
    directory_path, line_before_elec_str="PAW_PBE", potcar_filename="POTCAR"
):

    structure = _try_read_structure(directory_path=directory_path)
    if structure != None:
        element_list, element_count = element_count_ordered(structure)

    electron_of_potcar = []

    with open(os.path.join(directory_path, potcar_filename), "r") as file:
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


def get_total_electron_count(
    directory_path, line_before_elec_str="PAW_PBE", potcar_filename="POTCAR"
):
    ele_list, ele_count, electron_of_potcar = grab_electron_info(
        directory_path=directory_path,
        line_before_elec_str=line_before_elec_str,
        potcar_filename=potcar_filename,
    )
    total_electron_count = np.dot(ele_count, electron_of_potcar)
    return total_electron_count


def _check_convergence(directory):
    return directory, check_convergence(directory)


def find_converged_dirs(parent_dir):

    dirs = find_vasp_directories(parent_dir=parent_dir, extract_tarballs=False)
    # Filter the directories where convergence is True
    dir_and_convergence = parallelise(_check_convergence, dirs)

    converged_dirs = [
        directory for directory, convergence in dir_and_convergence if convergence
    ]
    return converged_dirs


def flatten_all_iterables(input_list):
    flat_list = []
    for item in input_list:
        if isinstance(
            item, (list, tuple, np.ndarray)
        ):  # Now also checks for numpy arrays
            flat_list.extend(item)  # Extend the flat list with elements of the iterable
        else:
            flat_list.append(item)  # Add the item directly if it's not an iterable
    return flat_list


def find_significantly_different_indices_threshold(values, threshold):
    if not values:
        return []
    significant_indices = [0]
    last_significant_value = values[0]
    for i, current_value in enumerate(values[1:], start=1):
        if abs(current_value - last_significant_value) >= threshold:
            significant_indices.append(i)
            last_significant_value = current_value
    return significant_indices


def exclude_non_converged_data(df, columns_to_exclude_data):
    def process_row(row):
        non_converged_indices = [
            i for i, conv in enumerate(row["scf_convergence"]) if not conv
        ]
        for column in columns_to_exclude_data:
            if column in row:  # Check if column is in row to avoid KeyError
                row[column] = [
                    value
                    for i, value in enumerate(row[column])
                    if i not in non_converged_indices
                ]
        return row

    processed_df = df.apply(process_row, axis=1)
    return processed_df


def get_flattened_df(
    df,
    groupby="filepath",
    columns_to_process=[
        "energy",
        "energy_zero",
        "structures",
        "forces",
        "magmoms",
        "stresses",
        "scf_steps",
        "scf_convergence",
    ],
):
    processed_df = (
        df.sort_values("calc_start_time")
        .groupby(groupby)
        .agg(lambda x: x.tolist())
        .reset_index()
        .copy()
    )
    for column in columns_to_process:
        processed_df[column] = processed_df[column].apply(flatten_all_iterables)
    return processed_df


def get_filtered_df(
    df,
    energy_threshold=0.05,
    columns=[
        "energy",
        "energy_zero",
        "structures",
        "forces",
        "magmoms",
        "stresses",
        "scf_steps",
        "scf_convergence",
    ],
):
    def process_row(row, column="energy", columns_to_flatten=columns):
        indices = find_significantly_different_indices_threshold(
            row[column], energy_threshold
        )
        processed_row = {
            col: (
                row[col]
                if col not in columns_to_flatten
                else [row[col][i] for i in indices if i < len(row[col])]
            )
            for col in df.columns
        }
        return processed_row

    significant_changes = df.apply(process_row, axis=1)
    significant_changes_df = pd.DataFrame(list(significant_changes))

    if "job_name" in significant_changes_df.columns:
        significant_changes_df["job_name"] = [
            row.job_name[0] for _, row in significant_changes_df.iterrows()
        ]

    return significant_changes_df


def get_potential_data_df(
    df,
    energy_threshold=0.05,
    columns_to_process=[
        "energy",
        "energy_zero",
        "structures",
        "forces",
        "magmoms",
        "stresses",
        "scf_steps",
        "scf_convergence",
    ],
):
    processed_df = get_flattened_df(df)
    processed_df = get_filtered_df(
        processed_df,
        energy_threshold=energy_threshold,
        columns=[
            "energy",
            "energy_zero",
            "structures",
            "forces",
            "magmoms",
            "stresses",
            "scf_steps",
            "scf_convergence",
        ],
    )
    non_corr_df = exclude_non_converged_data(processed_df, columns_to_process)

    return non_corr_df


class DatabaseGenerator:

    def __init__(self, parent_dir, max_workers=16):
        self.parent_dir = parent_dir
        self.max_workers = max_workers

    def build_database(
        self,
        target_directory=None,
        extract_directories=False,
        tarball_extensions=(".tar.gz", "tar.bz2"),
        read_error_dirs=False,
        read_multiple_runs_in_dir=False,
        cleanup=False,
        keep_filenames_after_cleanup=[],
        keep_filename_patterns_after_cleanup=[],
        max_dir_count=None,
        filenames_to_qualify=[
            "vasp.log",
            "INCAR",
            "POTCAR",
            "CONTCAR",
            "KPOINTS",
            "OUTCAR",
            "vasprun.xml",
        ],
        all_present=False,
        df_filename=None,
        df_compression=True,
    ):  # Added database_compression flag with default True

        start_time = time.time()

        if target_directory:
            dirs = find_vasp_directories(
                parent_dir=target_directory,
                extract_tarballs=extract_directories,
                all_present=all_present,
                filenames=filenames_to_qualify,
                tarball_extensions=tarball_extensions,
            )
        else:
            dirs = find_vasp_directories(
                parent_dir=self.parent_dir,
                extract_tarballs=extract_directories,
                all_present=all_present,
                filenames=filenames_to_qualify,
                tarball_extensions=tarball_extensions,
            )
        print(
            f"The total number of vasp directories that we are building the database out of is {len(dirs)}"
        )

        compression_option = "gzip" if df_compression else None
        compression_extension = ".gz" if df_compression else ""

        if max_dir_count:
            pkl_filenames = []
            for i, chunks in enumerate(gen_tools.chunk_list(dirs, max_dir_count)):
                step_time = time.time()
                df = pd.concat(
                    parallelise(
                        parse_vasp_directory,
                        [(chunk,) for chunk in chunks],
                        max_workers=self.max_workers,
                        extract_error_dirs=read_error_dirs,
                        parse_all_in_dir=read_multiple_runs_in_dir,
                    )
                )
                if df_filename:
                    db_filename = f"{i}_{df_filename}.pkl{compression_extension}"
                else:
                    db_filename = f"{i}.pkl{compression_extension}"
                pkl_filenames.append(os.path.join(self.parent_dir, db_filename))
                df.to_pickle(
                    os.path.join(self.parent_dir, db_filename),
                    compression=compression_option,
                )
                step_taken_time = np.round(time.time() - step_time, 3)
                print(
                    f"Step {i}: {step_taken_time} seconds taken for {len(chunks)} parse steps"
                )

            df = pd.concat(
                [
                    pd.read_pickle(partial_df, compression=compression_option)
                    for partial_df in pkl_filenames
                ]
            )
            final_db_filename = os.path.join(
                self.parent_dir, f"vasp_database.pkl{compression_extension}"
            )
            df.to_pickle(final_db_filename, compression=compression_option)
        else:
            df = pd.concat(
                parallelise(
                    parse_vasp_directory,
                    [(chunk,) for chunk in chunks],
                    max_workers=self.max_workers,
                    extract_error_dirs=read_error_dirs,
                    parse_all_in_dir=read_multiple_runs_in_dir,
                )
            )
            df.to_pickle(
                os.path.join(
                    self.parent_dir, f"vasp_database.pkl{compression_extension}"
                ),
                compression=compression_option,
            )

        end_time = time.time()
        elapsed_time = end_time - start_time

        # not optional - keep the tarballs/zips..
        keep_filename_patterns_after_cleanup += ".tar.gz"
        keep_filename_patterns_after_cleanup += ".tar.bz2"
        keep_filename_patterns_after_cleanup += ".zip"

        if cleanup:
            gen_tools.cleanup_dir(
                directory_path=dirs, keep=True, files=[], file_patterns=[]
            )
            parallelise(
                gen_tools.cleanup_dir,
                dirs,
                [True] * len(dirs),
                keep_filenames_after_cleanup * len(dirs),
                keep_filename_patterns_after_cleanup * len(dirs),
            )

        print("Elapsed time:", np.round(elapsed_time, 3), "seconds")

        return df

    def update_failed_jobs_in_database(
        self,
        df_path=None,
        read_error_dirs=False,
        read_multiple_runs_in_dir=False,
        max_dir_count=None,
        df_compression=True,
    ):
        compression_option = "gzip" if df_compression else None
        compression_extension = ".gz" if df_compression else ""

        if df_path is None:
            df_path = os.path.join(
                self.parent_dir, f"vasp_database.pkl{compression_extension}"
            )

        if os.path.isdir(df_path):
            potential_files = [
                os.path.join(df_path, "vasp_database.pkl.gz"),
                os.path.join(df_path, "vasp_database.pkl"),
            ]
            output_path = os.path.join(
                df_path, f"vasp_database.pkl{compression_extension}"
            )
        else:
            potential_files = [df_path]
            output_path = df_path

        df = None
        for file in potential_files:
            try:
                if file.endswith(".gz"):
                    df = pd.read_pickle(file, compression="gzip")
                else:
                    df = pd.read_pickle(file, compression=None)
                print(f"Successfully read database from {file}")
                break
            except (FileNotFoundError, pd.errors.UnrecognizedCompressionError):
                print(f"Failed to read database from {file}")

        if df is None:
            raise ValueError(
                "Invalid path or filename - please check! Attempted paths: "
                + ", ".join(potential_files)
            )

        failed_dirs = df[df["convergence"] == False]["filepath"].tolist()
        print(f"Reparsing {len(failed_dirs)} directories where convergence is False")

        if max_dir_count:
            pkl_filenames = []
            for i, chunks in enumerate(
                gen_tools.chunk_list(failed_dirs, max_dir_count)
            ):
                step_time = time.time()
                failed_df = pd.concat(
                    parallelise(
                        parse_vasp_directory,
                        [(chunk,) for chunk in chunks],
                        max_workers=self.max_workers,
                        extract_error_dirs=read_error_dirs,
                        parse_all_in_dir=read_multiple_runs_in_dir,
                    )
                )
                db_filename = f"update_{i}.pkl{compression_extension}"
                pkl_filenames.append(os.path.join(self.parent_dir, db_filename))
                failed_df.to_pickle(
                    os.path.join(self.parent_dir, db_filename),
                    compression=compression_option,
                )
                step_taken_time = np.round(time.time() - step_time, 3)
                print(
                    f"Step {i}: {step_taken_time} seconds taken for {len(chunks)} parse steps"
                )

            failed_df = pd.concat(
                [
                    pd.read_pickle(partial_df, compression=compression_option)
                    for partial_df in pkl_filenames
                ]
            )
        else:
            failed_df = pd.concat(
                parallelise(
                    parse_vasp_directory,
                    [(chunk,) for chunk in failed_dirs],
                    max_workers=self.max_workers,
                    extract_error_dirs=read_error_dirs,
                    parse_all_in_dir=read_multiple_runs_in_dir,
                )
            )

        # Use a different method to merge the DataFrames
        df.update(failed_df, overwrite=True)

        df.to_pickle(output_path, compression=compression_option)
        print(f"Updated dataframe saved to {output_path}")
        return df

    def build_potential_database(
        self,
        target_directory=None,
        extract_directories=False,
        tarball_extensions=(".tar.gz", "tar.bz2"),
        read_error_dirs=False,
        read_multiple_runs_in_dir=False,
        cleanup=False,
        keep_filenames_after_cleanup=[],
        keep_filename_patterns_after_cleanup=[],
        max_dir_count=None,
        filenames_to_qualify=[
            "vasp.log",
            "INCAR",
            "POTCAR",
            "CONTCAR",
            "KPOINTS",
            "OUTCAR",
            "vasprun.xml",
        ],
        all_present=False,
        df_filename=None,
        df_compression=True,
        incar_checks={},
        energy_threshold=0.1  # eV/atom
    ):

        def drop_nan_structures_rows(df, column_name="structures"):
            return df[~df[column_name].apply(lambda x: isinstance(x, list) and all(pd.isna(y) for y in x))].reset_index(drop=True)

        def unfold_rows(df):
            def unfold_row(row):
                num_elements = len(row['structures'])
                return [{col: (row[col][i] if isinstance(row[col], (list, np.ndarray)) and i < len(row[col]) else row[col]) for col in df.columns} for i in range(num_elements)]
            return pd.DataFrame([item for sublist in df.apply(unfold_row, axis=1) for item in sublist])

        def check_INCAR_params(df, check_pairs):
            def _check_INCAR_param(incar):
                return all(incar.get(key) == value for key, value in check_pairs.items()) if isinstance(incar, dict) else False
            df['INCAR_ok'] = df['INCAR'].apply(_check_INCAR_param)
            return df
        
        def filter_by_ev_atom(df, column="eV_atom", threshold=0.1):
            df_sorted = df.sort_values(by=column).reset_index(drop=True)
            filtered_rows = []

            last_retained_value = None
            for index, row in df_sorted.iterrows():
                if last_retained_value is None or abs(row[column] - last_retained_value) > threshold:
                    filtered_rows.append(row)
                    last_retained_value = row[column]
            
            return pd.DataFrame(filtered_rows)

        def apply_filter_to_groups(df, group_column="job_name", filter_column="eV_atom", threshold=0.1):
            filtered_dfs = df.groupby(group_column).apply(
                lambda group: filter_by_ev_atom(group, column=filter_column, threshold=threshold)
            )
            filtered_dfs = filtered_dfs.reset_index(drop=True)
            return filtered_dfs
        
        df = self.build_database(
            target_directory=target_directory,
            extract_directories=extract_directories,
            tarball_extensions=tarball_extensions,
            read_error_dirs=read_error_dirs,
            read_multiple_runs_in_dir=read_multiple_runs_in_dir,
            cleanup=cleanup,
            keep_filenames_after_cleanup=keep_filenames_after_cleanup,
            keep_filename_patterns_after_cleanup=keep_filename_patterns_after_cleanup,
            max_dir_count=max_dir_count,
            filenames_to_qualify=filenames_to_qualify,
            all_present=all_present,
            df_filename=df_filename,
            df_compression=df_compression
        )

        df = drop_nan_structures_rows(df)
        df = df[["job_name", "INCAR", "structures", "energy", "forces", "stresses", "magmoms"]]
        df = unfold_rows(df)
        df = check_INCAR_params(df, incar_checks)
        df = df[(df["INCAR_ok"]) | (df['INCAR'].notnull())]
        
        df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
        df = df.dropna(subset=['energy'])
        df["structures"] = df.structures.apply(lambda x: Structure.from_str(x, fmt="json"))
        df["n_atoms"] = df.structures.apply(lambda x: x.num_sites)
        df["eV_atom"] = df.energy/df.n_atoms

        df = apply_filter_to_groups(df, group_column="job_name", filter_column="eV_atom", threshold=energy_threshold)

        return df
    # def update_database(self,
    #                 new_calculation_directory,
    #                 existing_database_filename = "vasp_database.pkl",
    #                 extract_directories = True,
    #                 cleanup=False,
    #                 keep_filenames_after_cleanup = [],
    #                 keep_filename_patterns_after_cleanup = [],
    #                 max_dir_count = None,
    #                 df_filename = None):

    #     update_df = self.build_database(target_directory = existing_database_filename,
    #                                     extract_directories = extract_directories,
    #                                     cleanup=cleanup,
    #                                     keep_filenames_after_cleanup = keep_filenames_after_cleanup,
    #                                     keep_filename_patterns_after_cleanup = keep_filename_patterns_after_cleanup,
    #                                     max_dir_count = max_dir_count,
    #                                     df_filename = df_filename)
    #     def _get_job_dir(filepath):
    #         return os.path.basename(filepath.rstrip("/OUTCAR"))

    #     update_df["job_dir"] = [_get_job_dir(row.filepath) for _, row in update_df.iterrows()]
    #     base_df["job_dir"] = [_get_job_dir(row.filepath) for _, row in base_df.iterrows()]

    #     base_df = pd.read_pickle(existing_database_filename)

    #     # Merge df1 and df2 based on the common dirname
    #     interm_df = base_df.merge(update_df, on='job_dir', suffixes=('_df1', '_df2'), how='left')

    #     # Loop through the columns and update them dynamically
    #     for column in base_df.columns:
    #         if column not in ('filepath', 'job_dir'):
    #             # Check if the column with suffix '_df2' exists
    #             if (f'{column}_df2' in interm_df.columns):
    #                 base_df[column].update(interm_df[column + '_df2'].combine_first(interm_df[column + '_df1']))

    #     base_df.drop(columns=['job_dir'], inplace=True)

    #     return base_df


def update_database(df_base, df_update):
    # Get the unique job names from df2
    df_update_jobs = set(df_update["job_name"])

    # Filter rows in df1 that have job names not present in df2
    df_base = df_base[~df_base["job_name"].isin(df_update_jobs)].copy()

    # Append df2 to the filtered df1
    merged_df = pd.concat([df_base, df_update_jobs], ignore_index=True)
    return merged_df


def robust_append_last(clist, value):
    try:
        clist.append(value[-1])
    except (IndexError, TypeError):
        clist.append(np.nan)
    return clist
    
def create_summary(database_df):
    energies = []
    magmoms = []
    structures = []

    for i, row in database_df.iterrows():
        energies = robust_append_last(energies, row.energy_zero)
        magmoms = robust_append_last(magmoms, row.magmoms)
        structures = robust_append_last(structures, row.structures)

    df = database_df[["job_name", "convergence"]].copy()
    df["total_energy"] = energies
    df["magmoms"] = magmoms
    df["structures"] = structures
    return df
