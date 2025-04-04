from utils.potential_df_adapters.general import voigt_to_tensor, tensor_to_voigt
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np

def get_ace_fitting_df(potential_df,
                       energy_offset_per_atom=0, 
                       filter_lreal=False, 
                       lreal_value=False, 
                       force_threshold=10, 
                       eV_atom_min=-10, 
                       eV_atom_max=5):
    """
    Process and filter a DataFrame for ACE fitting with additional force and energy filters.
    
    Parameters:
    df_path (str): Path to the DataFrame pickle file.
    energy_offset_per_atom (float): Energy offset per atom to apply in correction.
    filter_lreal (bool): Whether to filter based on the 'LREAL' INCAR parameter.
    lreal_value (bool): Value of 'LREAL' to filter on if filter_lreal is True.
    force_threshold (float): Maximum allowable force for filtering.
    eV_atom_min (float): Minimum allowable energy per atom for filtering.
    eV_atom_max (float): Maximum allowable energy per atom for filtering.
    
    Returns:
    pd.DataFrame: Processed and filtered DataFrame for ACE fitting.
    """

    # Select relevant columns and filter based on convergence
    ace_df = potential_df.copy()[["job_name", "energy", "forces", "structure", "stresses", "magmoms", "scf_convergence", "incar"]]
    ace_df = ace_df[ace_df["scf_convergence"] == True]

    # Apply transformations and corrections
    ace_df["stresses"] = ace_df.stresses.apply(lambda x: -tensor_to_voigt(x / 1602.1766208))
    ace_df["forces"] = ace_df.forces.apply(lambda x: x)
    ace_df["energy"] = ace_df.energy.apply(lambda x: x)
    ace_df["LREAL"] = ace_df.incar.apply(lambda x: x["LREAL"])
    ace_df["structure"] = ace_df.structure.apply(lambda x: x)
    ace_df["num_sites"] = ace_df.structure.apply(lambda x: x.num_sites)
    ace_df["structure"] = ace_df.structure.apply(lambda x: AseAtomsAdaptor.get_atoms(x))
    ace_df["energy_corrected"] = ace_df["energy"] - (ace_df["num_sites"] * energy_offset_per_atom)
    ace_df = ace_df.rename(columns={"structure": "ase_atoms"})
    
    # Calculate max force per cell and add it as a new column
    ace_df['max_force'] = ace_df['forces'].apply(lambda x: np.max(np.linalg.norm(x, axis=1)))
    
    # Apply filtering on 'LREAL' if requested
    if filter_lreal:
        ace_df = ace_df[ace_df["LREAL"] == lreal_value]
    
    # Apply force and energy filtering
    ace_df = ace_df[ace_df['max_force'] < force_threshold]  # Filter by max force
    ace_df["eV_atom"] = ace_df["energy_corrected"] / ace_df["num_sites"]  # Compute eV/atom
    ace_df = ace_df[(ace_df['eV_atom'] <= eV_atom_max) & (ace_df['eV_atom'] > eV_atom_min)]  # Filter by eV/atom range

    return ace_df