import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.vasp.database import create_summary
from pymatgen.core import Structure
from joblib import Parallel, delayed
from tqdm import tqdm

from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from utils.structure_featuriser import VoronoiSiteFeaturiser
import numpy as np
import pandas as pd

def tensor_to_voigt(tensor):
    """
    Convert a 3x3 tensor to a 6-component Voigt notation vector.
    
    Parameters:
    tensor (numpy.ndarray): A 3x3 matrix representing the tensor.
    
    Returns:
    numpy.ndarray: A 6-component vector in Voigt notation.
    """
    if tensor.shape != (3, 3):
        raise ValueError("Input must be a 3x3 tensor")
    
    voigt = np.zeros(6)
    voigt[0] = tensor[0, 0]
    voigt[1] = tensor[1, 1]
    voigt[2] = tensor[2, 2]
    voigt[3] = tensor[1, 2]  # or tensor[2, 1]
    voigt[4] = tensor[0, 2]  # or tensor[2, 0]
    voigt[5] = tensor[0, 1]  # or tensor[1, 0]
    
    return voigt

def voigt_to_tensor(voigt):
    """
    Convert a 6-component Voigt notation vector to a 3x3 tensor.
    
    Parameters:
    voigt (numpy.ndarray): A 6-component vector in Voigt notation.
    
    Returns:
    numpy.ndarray: A 3x3 matrix representing the tensor.
    """
    if voigt.shape != (6,):
        raise ValueError("Input must be a 6-component vector")
    
    tensor = np.zeros((3, 3))
    tensor[0, 0] = voigt[0]
    tensor[1, 1] = voigt[1]
    tensor[2, 2] = voigt[2]
    tensor[1, 2] = voigt[3]
    tensor[2, 1] = voigt[3]
    tensor[0, 2] = voigt[4]
    tensor[2, 0] = voigt[4]
    tensor[0, 1] = voigt[5]
    tensor[1, 0] = voigt[5]
    
    return tensor

def get_robust_last(lst, default=np.nan):
    """
    Returns the last valid element of a list.
    If the list is empty or an error occurs, returns the default value.
    
    Parameters:
    - lst: The list from which to retrieve the last element.
    - default: The default value to return if the list is empty.
    
    Returns:
    - The last valid element or the default value.
    """
    try:
        if len(lst) > 0:
            return lst[-1]
        else:
            return default
    except Exception as e:
        #print(f"Error retrieving the last element: {e}")
        return default

def get_potential_df(df, clean=True):
    """
    Processes the input DataFrame and extracts energy, forces, stresses, magnetic moments, 
    INCAR parameters, KPOINTS, SCF convergence, and job name for each row.
    
    Parameters:
    - df: Input DataFrame with the required columns.
    
    Returns:
    - training_df: DataFrame containing the extracted data.
    """
    if clean:
        df = clean_df(df)
    # Initialize lists to hold the extracted data
    energy_lst = []
    forces_lst = []
    stresses_lst = []
    magmoms_lst = []
    incar_lst = []
    kspacing_lst = []
    scf_lst = []
    jobname_lst = []
    structure_lst = []
    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        # Extract data for each row
        try:
            structure = Structure.from_str(get_robust_last(row.get('structures', None)), fmt="json")
        except:
            structure = np.nan
        energy = get_robust_last(row.get('energy', None))
        forces = get_robust_last(row.get('forces', None))
        stresses = get_robust_last(row.get('stresses', None))
        magmoms = get_robust_last(row.get('magmoms', None))
        incar = row.get('INCAR', None)
        kspacing = row.get('KPOINTS', None)
        scf_convergence = get_robust_last(row.get('scf_convergence', None))
        job_name = row.get('job_name', None)

        # Append the data to respective lists
        structure_lst.append(structure)
        energy_lst.append(energy)
        forces_lst.append(forces)
        stresses_lst.append(stresses)
        magmoms_lst.append(magmoms)
        incar_lst.append(incar)
        kspacing_lst.append(kspacing)
        scf_lst.append(scf_convergence)  # Append the correct SCF convergence value
        jobname_lst.append(job_name)

    # Create a new DataFrame from the lists
    training_df = pd.DataFrame({
        'job_name': jobname_lst,
        "structure": structure_lst,
        'energy': energy_lst,
        'forces': forces_lst,
        'stresses': stresses_lst,
        'magmoms': magmoms_lst,
        'incar': incar_lst,
        'kspacing': kspacing_lst,
        'scf_convergence': scf_lst,
    })
    training_df['max_force'] = training_df['forces'].apply(lambda x: np.max(np.linalg.norm(x, axis=1)))
    training_df["eV_atom"] = training_df["energy"] / training_df.structure.apply(lambda x: x.num_sites)

    return training_df

def clean_df(df, columns=None):
    df_treat = df.copy()
    """
    Cleans the DataFrame by dropping rows where the values in the specified columns
    are neither a non-empty NumPy array nor NaN.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to clean.
        columns (list of str, optional): List of column names to check.
            Defaults to ['energy', 'energy_zero', 'forces', 'stresses', 'magmoms'].
            
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    if columns is None:
        columns = ['energy', 'energy_zero', 'forces', 'stresses', 'magmoms']
        
    def is_valid(x):
        # If it's a NumPy array, check that it's not empty
        if isinstance(x, np.ndarray):
            return x.size > 0
        # Otherwise, allow NaN; if it's not NaN, it's invalid.
        return pd.isna(x)

    for col in columns:
        df_treat = df_treat.dropna(subset=[col])
        if col in df.columns:
            df_treat = df_treat[df_treat[col].apply(is_valid)]
        else:
            print(f"Warning: Column '{col}' not found in the DataFrame.")
    return df_treat

    
def filter_potential_df(potential_df,
                        min_eV_atom_threshold = None,
                        max_eV_atom_threshold = None,
                        max_force_threshold = None):
    return potential_df

def compute_voronoi_volumes(structure):
    """
    Computes Voronoi total volumes (VorNN_tot_vol) for all sites in a given structure.

    Parameters:
        structure: The structure object containing atomic sites.

    Returns:
        pd.Series: A pandas Series containing the VorNN_tot_vol for all sites.
    """
    df_struct = []

    # Iterate over each site in the structure
    for i, site in enumerate(structure):
        # Extract Voronoi features for the site
        df_str, df_prop = VoronoiSiteFeaturiser(structure, i)
        df_site = pd.DataFrame([df_prop], columns=df_str)  # Build a DataFrame for this site
        df_struct.append(df_site)

    # Concatenate all site DataFrames
    df_struct = pd.concat(df_struct, ignore_index=True)

    # Return the VorNN_tot_vol column
    return df_struct['VorNN_tot_vol'].values

# Function to apply compute_voronoi_volumes in parallel
def parallel_apply(data, func, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(delayed(func)(item) for item in tqdm(data))
    return results

def get_vorvols(potential_df):
    potential_df["VorVols"] = parallel_apply(potential_df.structure, compute_voronoi_volumes)
    return potential_df

