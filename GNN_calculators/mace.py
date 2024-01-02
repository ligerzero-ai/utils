from tqdm import tqdm
import time
import warnings

import numpy as np
import pandas as pd

from chgnet.model import CHGNet

import matgl
from matgl.ext.ase import M3GNetCalculator, MolecularDynamics, Relaxer
from pymatgen.io.ase import AseAtomsAdaptor

from mace.calculators import MACECalculator

warnings.simplefilter("ignore")

df = pd.read_pickle("unrel_df.pkl")

def calc_static_CHGNET(structure, chgnet=None):
    try:
        if chgnet is None:
            chgnet = CHGNet.load()
            print("model load occurred inside function calc_static")
        chgnet_pred = chgnet.predict_structure(structure)
        toten = float(chgnet_pred["e"]) * structure.num_sites
        forces = chgnet_pred["f"]
        magmoms = chgnet_pred["m"]
    except Exception as e:
        print(f"CHGNET evaluation failed with exception: {e} \n Probably the element you are trying does not exist in their dataset")
        return np.nan, np.nan, np.nan
    return toten, forces, magmoms

def calc_static_M3GNET(structure, m3gnet=None):
    try:
        if m3gnet is None:
            m3gnet = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        atoms = AseAtomsAdaptor().get_atoms(structure)
        atoms.set_calculator(M3GNetCalculator(m3gnet))
        toten = atoms.get_potential_energy()
        forces = atoms.get_forces()
    except Exception as e:
        print(f"M3GNET evaluation failed with exception: {e} \n Probably the element you are trying does not exist in their dataset")
        return np.nan, np.nan, np.nan
    return toten, forces, np.nan

def calc_static_MACE(structure, MACE="/g/data/v43/Han/mace/mace/calculators/foundations_models/2023-08-14-mace-universal.model", device="cpu", default_dtype="float32"):
    try:
        MACE_calculator = MACECalculator(
                                          model_paths=MACE,
                                          device=device,
                                          default_dtype=default_dtype,
                                        )
        atoms = AseAtomsAdaptor().get_atoms(structure)
        atoms.set_calculator(MACE_calculator)
        toten = atoms.get_potential_energy()
        forces = atoms.get_forces()
    except Exception as e:
        print(f"MACE evaluation failed with exception: {e} \n Probably the element you are trying does not exist in their dataset")
        return np.nan, np.nan, np.nan
    return toten, forces, np.nan

def calc_static_GNN(structure, model_type = None, model = None):
    if model_type == "mace":
        toten, forces, magmoms = calc_static_MACE(structure)
    elif model_type == "m3gnet":
        toten, forces, magmoms = calc_static_M3GNET(structure, m3gnet=model)
    elif model_type == "chgnet":
        toten, forces, magmoms = calc_static_CHGNET(structure, chgnet=model)
    else:
        warnings.warn(f"Specified model {model} is not a valid calculator, returning np.nan")
        toten = np.nan
        forces = np.nan
        magmoms = np.nan
    return toten, forces, magmoms
for model_type in ["chgnet"]:
    pureGB_toten_lst = []
    pureslab_toten_lst = []
    segGB_toten_lst = []
    solslab_toten_lst = []

    pureGB_f_lst = []
    pureslab_f_lst = []
    segGB_f_lst = []
    solslab_f_lst = []

    pureGB_m_lst = []
    pureslab_m_lst = []
    segGB_m_lst = []
    solslab_m_lst = []

    eseg_lst = []  # List to store calculated energy of segregation

    start_time = time.time()

    # Loading the model based on model_type
    if model_type == "chgnet":
        model = CHGNet.load()
    elif model_type == "m3gnet":
        model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    else:
        model = None

    i = 0

    # Wrap the outer loop with tqdm to add a progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Model: {model_type}"):
        i+=1
        #if i > 5:
        #    break
        pureGB_start = time.time()

        # Call the calc_static_GNN function
        pureGB_toten, pureGB_f, pureGB_m = calc_static_GNN(row.struct_pureGB, model_type=model_type, model=model)
        pureslab_toten, pureslab_f, pureslab_m = calc_static_GNN(row.struct_pureSLAB, model_type=model_type, model=model)
        segGB_toten, segGB_f, segGB_m = calc_static_GNN(row.struct_segGB, model_type=model_type, model=model)
        solslab_toten, solslab_f, solslab_m = calc_static_GNN(row.struct_solSLAB, model_type=model_type, model=model)

        # Append values to the corresponding lists
        pureGB_toten_lst.append(pureGB_toten)
        pureslab_toten_lst.append(pureslab_toten)
        segGB_toten_lst.append(segGB_toten)
        solslab_toten_lst.append(solslab_toten)

        pureGB_f_lst.append(pureGB_f)
        pureslab_f_lst.append(pureslab_f)
        segGB_f_lst.append(segGB_f)
        solslab_f_lst.append(solslab_f)

        pureGB_m_lst.append(pureGB_m)
        pureslab_m_lst.append(pureslab_m)
        segGB_m_lst.append(segGB_m)
        solslab_m_lst.append(solslab_m)

        # Calculate the energy of segregation at each step and append to the list
        eseg = (
            segGB_toten - pureGB_toten - (solslab_toten - pureslab_toten)
        )
        eseg_lst.append(eseg)
        print(f"{row.job_name}: Eseg = {eseg_lst[-1]}, DFT = {row.E_seg_DFT}, error_Eseg = {eseg_lst[-1] - row.E_seg_DFT}")
        print(f"Row processing time: {time.time() - pureGB_start:.4f} seconds")

    model_elapsed_time = time.time() - start_time
    print(f"Total time for model {model_type}: {model_elapsed_time:.4f} seconds")

    # Attach total energies to the DataFrame with suffix "_model"
    df[f"toten_pureGB_{model_type}"] = pureGB_toten_lst
    df[f"toten_pureSLAB_{model_type}"] = pureslab_toten_lst
    df[f"toten_segGB_{model_type}"] = segGB_toten_lst
    df[f"toten_solSLAB_{model_type}"] = solslab_toten_lst

    # Attach individual components to the DataFrame with suffix "_model"
    df[f"pureGB_f_{model_type}"] = pureGB_f_lst
    df[f"pureslab_f_{model_type}"] = pureslab_f_lst
    df[f"segGB_f_{model_type}"] = segGB_f_lst
    df[f"solslab_f_{model_type}"] = solslab_f_lst

    df[f"pureGB_m_{model_type}"] = pureGB_m_lst
    df[f"pureslab_m_{model_type}"] = pureslab_m_lst
    df[f"segGB_m_{model_type}"] = segGB_m_lst
    df[f"solslab_m_{model_type}"] = solslab_m_lst

    #df.to_pickle(f"df_{model_type}.pkl")

    # Attach energy of segregation to the DataFrame with the corresponding suffix
    df[f"eseg_{model_type}"] = eseg_lst
df.to_pickle(f"df_{model_type}.pkl")
