import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from maml.describers import SmoothOverlapAtomicPosition

from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Structure

def get_stats(property_list, property_str):
    return  [f"{property_str}_std",f"{property_str}_mean",f"{property_str}_min",f"{property_str}_max"],\
            [np.std(property_list), np.mean(property_list), np.min(property_list), np.max(property_list)]
            
def VoronoiSiteFeaturiser(structure, site):
    coord_no = VoronoiNN().get_cn(structure = structure, n = site)
    site_info_dict = VoronoiNN().get_voronoi_polyhedra(structure, site)
    volumes = [site_info_dict[polyhedra]["volume"] for polyhedra in list(site_info_dict.keys())]
    vertices = [site_info_dict[polyhedra]["n_verts"] for polyhedra in list(site_info_dict.keys())]
    distances = [site_info_dict[polyhedra]["face_dist"] for polyhedra in list(site_info_dict.keys())]
    areas = [site_info_dict[polyhedra]["area"] for polyhedra in list(site_info_dict.keys())]
    
    total_area = np.sum(areas)
    total_volume = np.sum(volumes)
    
    df_str_list = ["VorNN_CoordNo", "VorNN_tot_vol", "VorNN_tot_area"]
    df_prop_list = [coord_no, total_volume, total_area]
    
    data_str_list = ["volumes", "vertices", "areas", "distances"]

    for i, value_list in enumerate([volumes, vertices, areas, distances]):
        property_str_list, property_stats_list = get_stats(value_list, f"VorNN_{data_str_list[i]}")
        df_str_list += property_str_list
        df_prop_list += property_stats_list
    
    return df_str_list, df_prop_list

def get_per_site_SOAP_descriptor(structure,
                                 cutoff=3,
                                 l_max=10,
                                 n_max=10,
                                 atom_sigma=0.5,
                                 verbose=False):
    """
    Process a list of pymatgen structures using the Smooth Overlap of Atomic Positions (SOAP) method
    and organize the results into a list of DataFrames per structure,
    with each row corresponding to the SOAP vector for that site.

    Parameters:
        struct_list (list): List of structures to process using SOAP.
        cutoff (float): Cutoff radius for the SOAP descriptor.
        l_max (int): Maximum number of angular momentum indices.
        n_max (int): Maximum number of radial indices.
        atom_sigma (float): Sigma value for atom type width in SOAP. # I actually have no idea what this means
        verbose (bool): Whether to print verbose output during the process.
        n_jobs (int): Number of cores to use for parallel processing.

    Returns:
        list: List of DataFrames, each DataFrame containing the SOAP descriptors for each site in the structure.
    """
    # Change n_jobs to the number of cores you have available
    s = SmoothOverlapAtomicPosition(cutoff=cutoff, l_max=l_max, n_max=n_max, atom_sigma=atom_sigma, verbose=verbose, n_jobs=1)
    # Create a DataFrame with the list of structures
    df = pd.DataFrame({'structure': [structure]})
    # Transform the structures using SOAP
    a = s.transform(df["structure"])
    # Copy and reset the index of the transformed DataFrame
    soap_df = a.copy().reset_index()
    # Rename the "level_1" column to "site"
    soap_df.rename(columns={"level_1": "site"}, inplace=True)
    # Group the DataFrame by "input_index" and drop the "input_index" column from each group
    df_list = [soap_df.reset_index(drop=True).drop(columns='input_index') for _, soap_df in soap_df.groupby(["input_index"])]
    return df
def get_per_site_SOAP_dfs(struct_list, cutoff=3, l_max=10, n_max=10, atom_sigma=0.5, verbose=False, n_jobs=16):
    """
    Process a list of pymatgen structures using the Smooth Overlap of Atomic Positions (SOAP) method
    and organize the results into a list of DataFrames per structure,
    with each row corresponding to the SOAP vector for that site.

    Parameters:
        struct_list (list): List of structures to process using SOAP.
        cutoff (float): Cutoff radius for the SOAP descriptor.
        l_max (int): Maximum number of angular momentum indices.
        n_max (int): Maximum number of radial indices.
        atom_sigma (float): Sigma value for atom type width in SOAP. # I actually have no idea what this means
        verbose (bool): Whether to print verbose output during the process.
        n_jobs (int): Number of cores to use for parallel processing.

    Returns:
        list: List of DataFrames, each DataFrame containing the SOAP descriptors for each site in the structure.
    """
    # Change n_jobs to the number of cores you have available
    s = SmoothOverlapAtomicPosition(cutoff=cutoff, l_max=l_max, n_max=n_max, atom_sigma=atom_sigma, verbose=verbose, n_jobs=n_jobs)
    # Create a DataFrame with the list of structures
    df = pd.DataFrame({'structure': struct_list})
    # Transform the structures using SOAP
    a = s.transform(df["structure"])
    # Copy and reset the index of the transformed DataFrame
    soap_df = a.copy().reset_index()
    # Rename the "level_1" column to "site"
    soap_df.rename(columns={"level_1": "site"}, inplace=True)
    # Group the DataFrame by "input_index" and drop the "input_index" column from each group
    df_list = [soap_df.reset_index(drop=True).drop(columns='input_index') for _, soap_df in soap_df.groupby(["input_index"])]

    return df_list

def get_SOAP_PCA_df(struct_list, PCA_comp = 30, write_df = False, filename=None):
    """
    Perform Principal Component Analysis (PCA) on Smooth Overlap of Atomic Positions (SOAP) descriptors
    for a list of structures and return a DataFrame with PCA-transformed SOAP descriptors.

    Parameters:
        struct_list (list): List of structures for which SOAP descriptors will be computed.
        PCA_comp (int, optional): Number of principal components to retain. Default is 30.
        write_df (bool, optional): Whether to save the PCA-transformed DataFrame as a pickle file.
        filename (str, optional): Name of the pickle file if `write_df` is True.

    Returns:
        pandas.DataFrame: DataFrame containing PCA-transformed SOAP descriptors with `PCA_comp` principal components.

    Note:
        This function internally calls `get_per_site_SOAP_df` to compute the SOAP descriptors for each site in the structures.
        The function concatenates all SOAP descriptors, performs standard scaling, and then applies PCA to obtain PCA-transformed SOAP descriptors.
        The DataFrame with PCA-transformed SOAP descriptors is returned, and it can be optionally saved as a pickle file if `write_df` is True and the `filename` is provided.
    """
    # Compute SOAP descriptors for each site in the structures
    struct_SOAP_df_list = get_per_site_SOAP_dfs(struct_list)
    
    # Concatenate all SOAP descriptors and perform standard scaling
    df_soap = pd.concat(struct_SOAP_df_list)
    df_soap.columns = df_soap.columns.astype(str)

    df_soap = StandardScaler().fit_transform(df_soap)
    # print("MEAN SHOULD BE ZERO, STD SHOULD BE 1")
    # print(np.mean(df_soap), np.std(df_soap))
    # Perform PCA with the specified number of principal components
    pca = PCA(n_components=PCA_comp)
    PCA_soap = pca.fit_transform(df_soap)
    
    # Create a DataFrame for PCA-transformed SOAP descriptors
    PCA_soap_df = pd.DataFrame(data=PCA_soap, columns=[f'SOAP_PCA_{i}' for i in np.arange(0, PCA_comp)])
    
    # Save the DataFrame as a pickle file if write_df is True
    if write_df:
        if filename:
            filename = filename
        else:
            filename = f"SOAP_PCA_{PCA_comp}_segsite.pkl"
        PCA_soap_df.to_pickle(filename)
        
    print(f'Explained variation at {PCA_comp} principal components: {np.sum(pca.explained_variance_ratio_)}')
    
    return PCA_soap_df

# def ACE_featuriser():