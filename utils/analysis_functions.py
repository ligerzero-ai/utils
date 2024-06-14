import os
import pandas as pd
import numpy as np
import shutil
import time
import re
import io
import json

from pymatgen.core import Structure
import pymatgen.transformations.site_transformations as transform
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core.lattice import Lattice
from pymatgen.core import Element

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Potcar, Incar, Kpoints
from pymatgen.io.vasp.outputs import Outcar

import matplotlib
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import utils.functions as func


# plt.rcParams['grid.color'] = 'k'
# plt.rcParams['grid.linestyle'] = ':'
# plt.rcParams['grid.linewidth'] = 0.5
# plt.rcParams.update({"font.family" : "Arial"})
# plt.rcParams.update({"font.size" : 20})

# GB legend


fp_plot_output = "C:\\Users\\liger\\OneDrive - The University of Sydney (Staff)\\FeGB-S-TM-Project\\Manuscript\\Figures"

my_dpi = 192
DataPath_FePtable = "E:\\Fe-Ptable-Data"
DataPath = "E:\\Fe-S-TM-Project"
DataPath = "D:\\Fe-SnAsSb-Tm-Project"
SetupPath = "C:\\Users\\liger\\Documents\\SetupPath\\FeGB-SnAsSb-TM-Project"

fp_Seg1_Krough = "%s\\Segregation-1sol-KPOINT_rough" % DataPath_FePtable
fp_Seg1_Kfine = "%s\\Segregation-1sol-KPOINT_fine" % DataPath_FePtable

fp_Seg2_Krough = "%s\\Segregation-2sol-Krough" % DataPath

job_script_folder = "/root/jobscript_templates"
job_path_DDEC6 = "%s\\jobfile-Gadi-DDEC6" % job_script_folder
job_path_standard_Gadi = "%s\\jobfile-conv-Gadi" % job_script_folder
job_path_standard_Magnus = "%s\\jobfile-conv-Magnus" % job_script_folder
job_path_standard_Setonix = "%s/jobfile-conv-Setonix" % job_script_folder
job_path_DoubleRelaxation_DDEC6_Setonix = os.path.join(
    job_script_folder, "jobfile-Setonix-DoubleRelaxation-DDEC6"
)
job_path_DoubleRelaxation_DDEC6_Gadi = os.path.join(
    job_script_folder, "jobfile-Gadi-DoubleRelaxation-DDEC6"
)
job_path_DoubleRelaxation_DDEC6_Gadi_GPU = os.path.join(
    job_script_folder, "jobfile-Gadi-DoubleRelaxation-DDEC6-GPU"
)
job_path_StaticImage_DDEC6_Setonix = os.path.join(
    job_script_folder, "jobfile-Setonix-StaticImage-DDEC6"
)
job_path_StaticImage_DDEC6_Gadi = os.path.join(
    job_script_folder, "jobfile-Gadi-StaticImage-DDEC6"
)

VASP_job_INCAR_path = "%s\\INCAR" % job_script_folder
VASP_job_INCAR_DDEC6_path = "%s\\INCAR-DDEC6" % job_script_folder

GB_Kpoint_rough_dict = {
    "S11-RA110-S3-32": func.KPOINTS([3, 3, 1], [0, 0, 0]),
    "S3-RA110-S1-11": func.KPOINTS([4, 2, 1], [0, 0, 0]),
    "S3-RA110-S1-12": func.KPOINTS([4, 3, 1], [0, 0, 0]),
    "S5-RA001-S210": func.KPOINTS([3, 3, 1], [0, 0, 0]),
    "S5-RA001-S310": func.KPOINTS([3, 2, 1], [0, 0, 0]),
    "S9-RA110-S2-21": func.KPOINTS([3, 4, 1], [0, 0, 0]),
}
# KPOINT NUMBER CALCULATED:  S3-1 6, S3-2 7, S9 7, S5-2 5, S5-3 4, S11 7
GB_Kpoint_fine_dict = {
    "S11-RA110-S3-32": func.KPOINTS([6, 6, 1], [0, 0, 0]),
    "S3-RA110-S1-11": func.KPOINTS([6, 3, 1], [0, 0, 0]),
    "S3-RA110-S1-12": func.KPOINTS([6, 6, 1], [0, 0, 0]),
    "S5-RA001-S210": func.KPOINTS([4, 4, 1], [0, 0, 0]),
    "S5-RA001-S310": func.KPOINTS([4, 3, 1], [0, 0, 0]),
    "S9-RA110-S2-21": func.KPOINTS([4, 6, 1], [0, 0, 0]),
}

list_GB = ["S11-RA110-S3-32", "S3-RA110-S1-11", "S3-RA110-S1-12", "S9-RA110-S2-21"]
list_element = ["P", "Ti", "V", "Cr", "Mn", "Co", "Ni", "Cu", "Nb", "Mo", "W"]


def get_immediate_subdirectories(a_dir):
    return [
        f.path
        for f in os.scandir(a_dir)
        if f.is_dir() and os.path.basename(f) != ".ipynb_checkpoints"
    ]


class DataPaths:
    def __init__(
        self,
        DataPath="C:\\Users\\liger\\OneDrive - The University of Sydney (Staff)\\FeGBProject-Data",
        Fpath_Krough="%s\\P-X-Krough" % DataPath,
        Fpath_Kfine="%s\\P-X-Krefined" % DataPath,
        Seg1_path="%s\\Segregation_1sol" % DataPath,
        Seg2_path="%s\\Segregation_2sol" % DataPath,
        Wsep1_rigid_path="%s\\RGS-1sol" % DataPath,
        Wsep1_rel_path="%s\\Wsep_relaxed-1sol" % DataPath,
        Wsep2_rigid_path="%s\\RGS" % DataPath,
        Wsep2_rel_path="%s\\Wsep_rel" % DataPath,
        BO1_path="%s\\BondOrder-1solute" % DataPath,
        BO2_path="%s\\BondOrder-2solute" % DataPath,
    ):

        self.DataPath = DataPath
        self.Fpath_Krough = "%s\\P-X-Krough" % DataPath
        self.Fpath_Kfine = "%s\\P-X-Krefined" % DataPath
        self.Seg1_path = "%s\\Segregation-1sol" % DataPath
        self.Seg2_path = "%s\\Segregation_2sol" % DataPath
        self.Wsep1_rigid_path = "%s\\RGS-1sol" % DataPath
        self.Wsep1_rel_path = "%s\\Wsep_relaxed-1sol" % DataPath
        self.Wsep2_rigid_path = "%s\\RGS" % DataPath
        self.Wsep2_rel_path = "%s\\Wsep_rel" % DataPath
        self.BO1_path = "%s\\BondOrder-1solute" % DataPath
        self.BO2_path = "%s\\BondOrder-2solute" % DataPath


class PlotParameters:
    """
    PlotParameters class contains object-based convenience functionality for plotting parameters
    """

    def __init__(
        self,
        output_path="C:\\Users\\liger\\OneDrive - The University of Sydney (Staff)\\FeGB-P-TM-Project\\Manuscript\\Figures-P-TM",
    ):
        self.marker_dict = dict(
            zip(
                [
                    "S3-RA110-S1-11",
                    "S3-RA110-S1-12",
                    "S9-RA110-S2-21",
                    "S11-RA110-S3-32",
                ],
                ["o", "s", "d", "^"],
            )
        )
        self.GB_labels = dict(
            zip(
                [
                    "S3-RA110-S1-11",
                    "S3-RA110-S1-12",
                    "S9-RA110-S2-21",
                    "S11-RA110-S3-32",
                ],
                [
                    r"$\Sigma3\ [110](1\bar{1}1)$",
                    r"$\Sigma3\ [110](1\bar{1}2)$",
                    r"$\Sigma9\ [110](2\bar{2}1)$",
                    r"$\Sigma11\ [110](3\bar{3}2)$",
                ],
            )
        )
        self.GB_labels_short = dict(
            zip(
                [
                    "S3-RA110-S1-11",
                    "S3-RA110-S1-12",
                    "S9-RA110-S2-21",
                    "S11-RA110-S3-32",
                ],
                [
                    r"$\Sigma3(1\bar{1}1)$",
                    r"$\Sigma3(1\bar{1}2)$",
                    r"$\Sigma9(2\bar{2}1)$",
                    r"$\Sigma11(3\bar{3}2)$",
                ],
            )
        )
        self.output_path = "C:\\Users\\liger\\OneDrive - The University of Sydney (Staff)\\FeGB-P-TM-Project\\Manuscript\\Figures-P-TM"

        self.label_string_S11_RA110_S3_32 = r"$\Sigma11\ [110](3\bar{3}2)$"
        self.label_string_S3_RA110_S1_11 = r"$\Sigma3\ [110](1\bar{1}1)$"
        self.label_string_S3_RA110_S1_12 = r"$\Sigma3\ [110](1\bar{1}2)$"
        self.label_string_S9_RA110_S2_21 = r"$\Sigma9\ [110](2\bar{2}1)$"
        self.color_ele_dict = dict(
            zip(
                ["P", "Ti", "V", "Cr", "Mn", "Co", "Ni", "Cu", "Nb", "Mo", "W"],
                [
                    "olive",
                    "blue",
                    "orange",
                    "green",
                    "red",
                    "black",
                    "brown",
                    "pink",
                    "darkviolet",
                    "lime",
                    "cyan",
                ],
            )
        )


class SegregationEnergyData_2sol:
    def __init__(self, savefile=True):

        Segregation_1sol = SegregationEnergyData_1sol()

        def get_1sol_cohesion_effect(GB, element, property, df=None):
            """
            GB = GB string (e.g. S11-RA110-S3-32)
            element = element string (e.g. V)
            property = one of string: eta_RGS, eta_rel, eta_ANSBO
            df = the output from get_1sol_cohesion_summary() method
            not strictly necessary, but is required for the purposes of speedup
            """
            if df is None:
                df = get_1sol_cohesion_summary(GB)
            ele_df = df[df["element"] == element]
            eta_1ele = ele_df[ele_df["E_seg"] == ele_df.E_seg.min()][property].values[0]
            return eta_1ele

        def get_2sol_cohesion_RGS(GB, case):

            GB_energetics_df = self.GB_treated_dfs_dict[GB]
            csv_path = "%s\\%s\\%s\\info.csv" % (fp_Wsep2_rigid, GB, case)
            if os.path.isfile(csv_path):
                area = get_area("%s\\%s\\Co\\GB\\CONTCAR" % (fp_Seg1_path, GB))
                case_df = pd.read_csv(csv_path)
                # total energy of non-cleaved GB structure
                total_energy = GB_energetics_df[
                    GB_energetics_df["system"] == case
                ].energy.values[0]
                Wsep_RGS_list = [
                    (row.energy - total_energy) * 16.02176565 / area
                    for _, row in case_df.iterrows()
                ]
                Wsep_RGS = min(np.array(Wsep_RGS_list))
                cp_list = [
                    float(row.system.split(sep="-")[-1])
                    for _, row in case_df.iterrows()
                ]
            else:
                Wsep_RGS = np.nan
                Wsep_RGS_list = np.nan
                cp_list = np.nan

            return Wsep_RGS, Wsep_RGS_list, cp_list

        def get_2sol_cohesion_Wseprel(GB, case):

            GB_energetics_df = self.GB_treated_dfs_dict[GB]
            csv_path = "%s\\%s\\info.csv" % (fp_Wsep2_rel, GB)
            df = pd.read_csv(csv_path)
            df["base_system"] = [
                "-".join(row.system.split(sep="-")[:-2]) for _, row in df.iterrows()
            ]

            if case in df.base_system.values:
                area = get_area("%s\\%s\\Co\\GB\\CONTCAR" % (fp_Seg1_path, GB))
                # total energy of non-cleaved GB structure
                total_energy = GB_energetics_df[
                    GB_energetics_df["system"] == case
                ].energy.values[0]
                Wsep_rel = (
                    (df[df["base_system"] == case].energy.values[0] - total_energy)
                    * 16.02176565
                    / area
                )
            else:
                Wsep_rel = np.nan

            return Wsep_rel

        def get_2sol_cohesion_BO(GB, case):
            csv_RGS_path = "%s\\%s\\%s\\info.csv" % (fp_Wsep2_rigid, GB, case)
            if os.path.isfile(csv_RGS_path) and os.path.isfile(
                "%s\\%s\\%s\\CONTCAR" % (fp_BO2, GB, case)
            ):
                case_df = pd.read_csv(csv_RGS_path)
                cp_list = [
                    float(row.system.split(sep="-")[-1])
                    for _, row in case_df.iterrows()
                ]
                min_bo_list, _ = cp_bondorder(
                    structure_path="%s\\%s\\%s\\CONTCAR" % (fp_BO2, GB, case),
                    DDEC_output_path="%s\\%s\\%s" % (fp_BO2, GB, case),
                    cleavage_plane_array=cp_list,
                    bo_threshold=0,
                )
                min_bo_CP = cp_list[np.argmin(min_bo_list)]
                min_bo = min(min_bo_list)
            else:
                min_bo = np.nan
                min_bo_list = np.nan
                min_bo_CP = np.nan
                cp_list = np.nan
            return min_bo, min_bo_list, min_bo_CP, cp_list

        dict_Krough_list = []
        for GB in get_immediate_subdirectories(fp_Fpath_Krough):
            df_Krough_list = []
            for element in get_immediate_subdirectories(GB):
                df_Krough = pd.read_csv("%s\\info.csv" % element)
                df_Krough["distance"] = [
                    get_dist_solutes("%s\\%s" % (element, row.system))
                    for i, row in df_Krough.iterrows()
                ]
                df_Krough["element"] = [
                    (
                        row.system.split(sep="-")[0]
                        if row.system.split(sep="-")[0] != "P"
                        else row.system.split(sep="-")[-2]
                    )
                    for i, row in df_Krough.iterrows()
                ]
                df_Krough_list.append(df_Krough)
            df_Krough_all = pd.concat(df_Krough_list)
            # Re-organise into ordering agnostic blocks (e.g. P-X and X-P both fall under "X" calls to dict)
            df_Krough_list = []
            for element in df_Krough_all.element.unique():
                df_Krough_list.append(
                    df_Krough_all[df_Krough_all["element"] == element]
                )
            df_Krough_dict = dict(zip(df_Krough_all.element.unique(), df_Krough_list))
            dict_Krough_list.append(df_Krough_dict)
        self.GB_Krough_df_dict = dict(
            zip(
                [
                    os.path.basename(GB)
                    for GB in get_immediate_subdirectories(fp_Fpath_Krough)
                ],
                dict_Krough_list,
            )
        )

        dict_Kfine_list = []
        for GB in get_immediate_subdirectories(fp_Fpath_Kfine):
            df_Kfine_list = []
            for element in get_immediate_subdirectories(GB):
                df_Kfine = pd.read_csv("%s\\info.csv" % element)
                df_Kfine["distance"] = [
                    get_dist_solutes("%s\\%s" % (element, row.system))
                    for i, row in df_Kfine.iterrows()
                ]
                df_Kfine["element"] = [
                    (
                        row.system.split(sep="-")[0]
                        if row.system.split(sep="-")[0] != "P"
                        else row.system.split(sep="-")[-2]
                    )
                    for i, row in df_Kfine.iterrows()
                ]
                df_Kfine_list.append(df_Kfine)
            df_Kfine_all = pd.concat(df_Kfine_list)

            df_Kfine_list = []
            for element in df_Kfine_all.element.unique():
                df_Kfine_list.append(df_Kfine_all[df_Kfine_all["element"] == element])
            df_Kfine_dict = dict(zip(df_Kfine_all.element.unique(), df_Kfine_list))
            dict_Kfine_list.append(df_Kfine_dict)
        self.GB_Kfine_df_dict = dict(
            zip(
                [
                    os.path.basename(GB)
                    for GB in get_immediate_subdirectories(fp_Fpath_Kfine)
                ],
                dict_Kfine_list,
            )
        )

        df_all_list = []
        for GB in self.GB_Kfine_df_dict:
            df_list = []
            for element in self.GB_Kfine_df_dict[GB]:
                if element in self.GB_Krough_df_dict[GB]:
                    df_Kr = self.GB_Krough_df_dict[GB][element].copy()
                    df_Kf = self.GB_Kfine_df_dict[GB][element].copy()
                    corr_list = []
                    for i, case in df_Kf.iterrows():
                        if len(df_Kr[df_Kr["system"] == case.system]):
                            energy_Kf = df_Kf[
                                df_Kf["system"] == case.system
                            ].energy.values[0]
                            energy_Kr = df_Kr[
                                df_Kr["system"] == case.system
                            ].energy.values[0]
                            # E_kr + corr = E_kf
                            correction = energy_Kf - energy_Kr
                            corr_list.append(correction)
                        else:
                            corr_list.append(np.nan)
                    df_Kr["energy"] = [
                        row.energy + np.mean(corr_list) for i, row in df_Kr.iterrows()
                    ]
                    df = df_Kf.append(df_Kr)
                    # print(corr_list)
                    # print("GB: %s, element: %s, corr_value: %.2f eV" % (GB, element, np.round(np.mean(corr_list),4)))
                    df.drop_duplicates(subset=["system"], keep="first", inplace=True)
                    df.sort_values("system", inplace=True)
                else:
                    df = self.GB_Kfine_df_dict[GB][element]
                df_list.append(df)
            GB_df = pd.concat(df_list)
            GB_df.sort_values("system", inplace=True)

            df_all_list.append(GB_df)
        self.GB_treated_dfs_dict = dict(zip(list_GB, df_all_list))

        GB_df_list = []
        for i, GB in enumerate(list_GB):
            GB_step_time = time.time()
            df = self.GB_treated_dfs_dict[GB].copy()
            df_1sol = get_1sol_cohesion_summary(GB)

            GB_pure_ANSBO = df_1sol[df_1sol["system"] == "GB"].ANSBO.values[0]
            GB_pure_WsepRGS = df_1sol[df_1sol["system"] == "GB"].Wsep_RGS.values[0]
            GB_pure_Wseprel = df_1sol[df_1sol["system"] == "GB"].Wsep_rel.values[0]

            df["element_1"] = [x.split(sep="-")[0] for x in df.system.values]
            df["element_2"] = [x.split(sep="-")[-2] for x in df.system.values]

            # Get the site number of solute 1
            df["site_1"] = [int(x.split(sep="-")[1]) for x in df.system.values]
            # Get the site number of solute 2
            df["site_2"] = [int(x.split(sep="-")[-1]) for x in df.system.values]
            # Get the Eseg in isolation of solute 1
            df["E_seg_s1"] = [
                Segregation_1sol.get_Eseg(
                    GB=GB, element=case.element_1, site=case.site_1
                )
                for i, case in df.iterrows()
            ]
            # Get the Eseg in isolation of solute 2
            df["E_seg_s2"] = [
                Segregation_1sol.get_Eseg(
                    GB=GB, element=case.element_2, site=case.site_2
                )
                for i, case in df.iterrows()
            ]
            if GB == "S11-RA110-S3-32":
                GB_use = "S11-RA110-S3-32-2x2"
            else:
                GB_use = GB
            # Get the energy of solution
            df["solnE_s1"] = [
                get_solution_energy(GB_use, case.element_1) for i, case in df.iterrows()
            ]
            df["solnE_s2"] = [
                get_solution_energy(GB_use, case.element_2) for i, case in df.iterrows()
            ]
            ## Get the energy of base solute structure in isolation of solute 1
            #### THIS NEEDS TO BE FIXED - TOTEN NEEDS TO BE THAT OF THE ORIGINAL SIZED GB (1x1 in S11, S9)
            # conditional_timer = time.time()
            E_base_list = []
            for i, case in df.iterrows():
                if GB == "S9-RA110-S2-21" and case.system.split(sep="-")[0] == "P":
                    # -549.38781758 is the value for P-36-d-0.0 structure in the 1x1 S9 cell
                    E_base_list.append(-549.38781758)
                else:
                    E_base_list.append(get_totalenergy(GB, structure_name=case.system))
            # print("%.2fs elapsed for checking conditional"% (time.time() - conditional_timer))
            df["E_base"] = E_base_list
            df["E_GB"] = get_totalenergy(GB)
            # Incremental energy of segregation
            df["E_seg_inc"] = df["energy"] - df["E_base"] - df["solnE_s2"]
            # IMPORTANT CORRECTION
            # We do not want positive energies of segregation - it has some meaning in that it means that further
            # segregation is super unfavourable within the studied zones of segregation (3-4A away from GB)
            # but the real incremental energy of segregation should be at infinite distance away from the GB, in which case
            # Eseg_inc = 0
            df["E_seg_inc_c"] = [x if x < 0 else 0 for x in df.E_seg_inc.values]
            # Total energy of segregation
            df["E_seg_total"] = (
                df["energy"] - df["E_GB"] - df["solnE_s2"] - df["solnE_s1"]
            )
            # This only works if the E_int is NOT POSITIVE
            df["E_correction"] = [0 if x < 0 else -x for x in df.E_seg_inc.values]
            # total energy with correction if Eseg_inc > 0
            df["toten_c"] = df["energy"] + df["E_correction"]
            # E_correction for applying to total energy.
            df["E_seg_total_c"] = (
                df["energy"]
                - df["E_GB"]
                - df["solnE_s2"]
                - df["solnE_s1"]
                + df["E_correction"]
            )
            # Interaction energy (If the first solute wasn't there, difference in energy)
            df["E_int"] = df["E_seg_inc"] - df["E_seg_s2"]

            #         # corrected energy of interaction
            #         df["E_int_c"] = [-float(get_1sol_EsegWithoutSite(os.path.basename(GB),x.element_2,x.site_1))\
            #                            if x.E_seg_inc_c == 0\
            #                            else x.E_seg_inc - x.E_seg_s2 for _, x in df.iterrows()]

            # df_out = df[["system","E_seg_inc", "E_int", "distance", "convergence"]]\

            df["Wsep_RGS"] = [
                get_2sol_cohesion_RGS(GB, row.system)[0] for _, row in df.iterrows()
            ]
            df["Wsep_RGS_list"] = [
                get_2sol_cohesion_RGS(GB, row.system)[1] for _, row in df.iterrows()
            ]
            df["cp_list"] = [
                get_2sol_cohesion_RGS(GB, row.system)[2] for _, row in df.iterrows()
            ]
            df["eta_RGS"] = df["Wsep_RGS"] - GB_pure_WsepRGS
            df["eta_RGS_ele1"] = [
                get_1sol_cohesion_effect(GB, row.element_1, "eta_RGS", df=df_1sol)
                for _, row in df.iterrows()
            ]
            df["eta_RGS_ele2"] = [
                get_1sol_cohesion_effect(GB, row.element_2, "eta_RGS", df=df_1sol)
                for _, row in df.iterrows()
            ]
            df["heur_eta_RGS"] = df["eta_RGS_ele1"] + df["eta_RGS_ele2"]

            df["Wsep_rel"] = [
                get_2sol_cohesion_Wseprel(GB, row.system) for _, row in df.iterrows()
            ]
            df["eta_rel"] = df["Wsep_rel"] - GB_pure_Wseprel
            df["eta_rel_ele1"] = [
                get_1sol_cohesion_effect(GB, row.element_1, "eta_rel", df=df_1sol)
                for _, row in df.iterrows()
            ]
            df["eta_rel_ele2"] = [
                get_1sol_cohesion_effect(GB, row.element_2, "eta_rel", df=df_1sol)
                for _, row in df.iterrows()
            ]
            df["heur_eta_rel"] = df["eta_rel_ele1"] + df["eta_rel_ele2"]

            df["ANSBO"] = [
                get_2sol_cohesion_BO(GB, row.system)[0] if row.E_seg_inc < 0 else np.nan
                for _, row in df.iterrows()
            ]
            df["ANSBO_list"] = [
                get_2sol_cohesion_BO(GB, row.system)[1] if row.E_seg_inc < 0 else np.nan
                for _, row in df.iterrows()
            ]
            df["ANSBO_min_cp"] = [
                get_2sol_cohesion_BO(GB, row.system)[2] if row.E_seg_inc < 0 else np.nan
                for _, row in df.iterrows()
            ]
            df["eta_ANSBO"] = df["ANSBO"] - GB_pure_ANSBO
            df["eta_ANSBO_ele1"] = [
                get_1sol_cohesion_effect(GB, row.element_1, "eta_ANSBO", df=df_1sol)
                for _, row in df.iterrows()
            ]
            df["eta_ANSBO_ele2"] = [
                get_1sol_cohesion_effect(GB, row.element_2, "eta_ANSBO", df=df_1sol)
                for _, row in df.iterrows()
            ]
            df["heur_eta_ANSBO"] = df["eta_ANSBO_ele1"] + df["eta_ANSBO_ele2"]

            df["element"] = [
                (
                    row.system.split(sep="-")[0]
                    if row.system.split(sep="-")[0] != "P"
                    else row.system.split(sep="-")[-2]
                )
                for i, row in df.iterrows()
            ]
            df_output = df.copy()[
                [
                    "system",
                    "element_1",
                    "site_1",
                    "element_2",
                    "site_2",
                    "energy",
                    "solnE_s1",
                    "solnE_s2",
                    "E_base",
                    "E_GB",
                    "E_seg_s1",
                    "E_seg_s2",
                    "E_seg_inc",
                    "E_seg_total_c",
                    "E_int",
                    "distance",
                    "element",
                    "Wsep_RGS",
                    "Wsep_rel",
                    "ANSBO",
                    "cp_list",
                    "Wsep_RGS_list",
                    "ANSBO_list",
                ]
            ]
            # df = np.round(df, 3)
            GB_df_list.append(df)
            print("%.2fs elapsed for GB step" % (time.time() - GB_step_time))
            if i == 0 and savefile:
                df_output.to_excel(
                    "%s\\energetics_analysis.xlsx" % (os.getcwd()),
                    sheet_name="%s" % (os.path.basename(GB)),
                )
            else:
                with pd.ExcelWriter(
                    "%s\\energetics_analysis.xlsx" % (os.getcwd()),
                    mode="a",
                    engine="openpyxl",
                ) as writer:
                    df_output.to_excel(writer, sheet_name="%s" % (os.path.basename(GB)))
        self.GB_energetics_dict = dict(zip(list_GB, GB_df_list))


class SegregationEnergyData_1sol:
    def __init__(self):
        ####################################################################################################
        # S3 S111
        studied_list = [20, 22, 24, 26, 28, 30, 32, 34, 36]
        # 0.5-1ML available
        symmetry = [
            [21, 52, 53],
            [23, 50, 51],
            [25, 48, 49],
            [27, 46, 47],
            [29, 44, 45],
            [31, 42, 43],
            [33, 40, 41],
            [35, 38, 39],
            [37],
        ]
        # When the site is on the GB plane, we don't need to calculate values on both sides
        self.S3_RA110_S1_11_symmetrydict = dict(zip(studied_list, symmetry))
        ####################################################################################################
        # S3 S112
        studied_list = [12, 14, 16, 18, 20, 22, 24]
        # 0.5-1ML available
        symmetry = [
            [13, 36, 37],
            [15, 34, 35],
            [17, 32, 33],
            [19, 30, 31],
            [21, 28, 29],
            [23, 26, 27],
            [25],
        ]
        # When the site is on the GB plane, we don't need to calculate values on both sides
        self.S3_RA110_S1_12_symmetrydict = dict(zip(studied_list, symmetry))
        ####################################################################################################
        # S9
        studied_list = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
        # only 0-1 ML available
        symmetry = [
            [47],
            [46],
            [45],
            [44],
            [43],
            [42],
            [41],
            [40],
            [39],
            [38],
            [37],
            [],
            [],
            [],
        ]
        # When the site is on the GB plane, we don't need to calculate values on both sides
        self.S9_RA110_S2_21_symmetrydict = dict(zip(studied_list, symmetry))
        ####################################################################################################
        # S11
        studied_list = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        # only 0-1 ML available
        symmetry = [[32], [31], [30], [29], [28], [27], [26], [25], [24], [23], [], []]
        # Full dictionary of solutes and sites
        self.S11_RA110_S3_32_symmetrydict = dict(zip(studied_list, symmetry))
        self.GB_sym_dict = dict(
            zip(
                list_GB,
                [
                    self.S11_RA110_S3_32_symmetrydict,
                    self.S3_RA110_S1_11_symmetrydict,
                    self.S3_RA110_S1_12_symmetrydict,
                    self.S9_RA110_S2_21_symmetrydict,
                ],
            )
        )
        ####################################################################################################
        # Create a dictionary that may be accessed
        # dict[GB][element][site] that contains information on all sites
        # (including symmetrically identical ones that were not computed)

        Eseg_data_list = []
        toten_data_list = []
        for GB in list_GB:
            sym_dict = self.GB_sym_dict[GB]

            if GB == "S11-RA110-S3-32":
                GB = "S11-RA110-S3-32-2x2"

            GB_path = "%s\\%s" % (fp_Seg1_path, GB)
            list_element = []
            ele_eseg_list = []
            ele_toten_list = []

            for element in get_immediate_subdirectories(GB_path):

                df = get_1sol_df("%s\\%s" % (GB_path, os.path.basename(element)))

                sym_df = df.copy()
                sym_df["base_site"] = [int(x) for x in sym_df["site"]]
                sym_df["site"] = [int(x.site) for i, x in sym_df.iterrows()]

                for _, sites_calculated in sym_df.iterrows():
                    for x in sym_dict[sites_calculated["base_site"]]:
                        sym_site = sites_calculated.copy()

                        sym_sys = sym_site["system"].split(sep="-")
                        sym_sys[1] = str(x)
                        sym_sys = "-".join(sym_sys)
                        sym_site["system"] = sym_sys

                        sym_site["site"] = int(x)
                        sym_df = sym_df.append(sym_site)

                ele_site_eseg_dict = dict(zip(sym_df.site, sym_df.E_seg.values))
                ele_site_toten_dict = dict(zip(sym_df.site, sym_df.energy.values))

                ele_eseg_list.append(ele_site_eseg_dict)
                ele_toten_list.append(ele_site_toten_dict)

                list_element.append(os.path.basename(element))

            ele_eseg_data_dict = dict(zip(list_element, ele_eseg_list))
            ele_toten_data_dict = dict(zip(list_element, ele_toten_list))

            Eseg_data_list.append(ele_eseg_data_dict)
            toten_data_list.append(ele_toten_data_dict)

        self.Eseg_dict = dict(zip(list_GB, Eseg_data_list))
        self.toten_dict = dict(zip(list_GB, toten_data_list))

    def get_Eseg(self, GB, element, site, warning=True):
        """
        Convenience value extractor for single solute segregation energy at specified GB, element, site
        Args:
        GB = string for GB (e.g. S11-RA110-S3-32)
        element = string for element (e.g. "W" for tungsten)
        site = integer for site (0-indexed)
        """
        if GB in self.Eseg_dict:
            if element in self.Eseg_dict[GB]:
                if site in self.Eseg_dict[GB][element]:
                    E_seg = self.Eseg_dict[GB][element][site]
                    # print(E_seg)
                else:
                    E_seg = np.nan
                    if warning:
                        print(
                            'No such site "%s" in element "%s" in %s dict: check site'
                            % (site, element, GB)
                        )
            else:
                E_seg = np.nan
                if warning:
                    print(
                        'No such element "%s" in %s dict: check element' % (element, GB)
                    )
        else:
            E_seg = np.nan
            if warning:
                print('No such GB "%s" in dict: check GB string' % GB)
        return E_seg

    def get_toten(self, GB, element, site, warning=True):
        """
        Convenience value extractor for single solute total energy at specified GB, element, site
        Args:
        GB = string for GB (e.g. S11-RA110-S3-32)
        element = string for element (e.g. "W" for tungsten)
        site = integer for site (0-indexed)
        """
        if GB in self.toten_dict:
            if element in self.toten_dict[GB]:
                if site in self.toten_dict[GB][element]:
                    toten = self.toten_dict[GB][element][site]
                    # print(E_seg)
                else:
                    toten = np.nan
                    if warning:
                        print(
                            'No such site "%s" in element "%s" in %s dict: check site'
                            % (site, element, GB)
                        )
            else:
                toten = np.nan
                if warning:
                    print(
                        'No such element "%s" in %s dict: check element' % (element, GB)
                    )
        else:
            toten = np.nan
            if warning:
                print('No such GB "%s" in dict: check GB string' % GB)
        return toten

    def get_min_Eseg_without_site(self, GB, element, site, warning=True):
        """
        Convenience value extractor for minimum single solute segregation energy at specified GB, element, without specified site
        Args:
        GB = string for GB (e.g. S11-RA110-S3-32)
        element = string for element (e.g. "W" for tungsten)
        site = integer for site to exclude (0-indexed)
        """
        if GB in self.Eseg_dict:
            if element in self.Eseg_dict[GB]:
                if site in self.Eseg_dict[GB][element]:
                    min_Eseg = self.Eseg_dict[GB][element][site]
                    # print(E_seg)
                else:
                    min_Eseg = np.nan
                    if warning:
                        print(
                            'No such site "%s" in element "%s" in %s dict: check site'
                            % (site, element, GB)
                        )
            else:
                min_Eseg = np.nan
                if warning:
                    print(
                        'No such element "%s" in %s dict: check element' % (element, GB)
                    )
        else:
            min_Eseg = np.nan
            if warning:
                print('No such GB "%s" in dict: check GB string' % GB)
        return toten


def get_dist_solutes(fp_2solutes):
    """
    Returns distance between pair of non-Fe solutes for a specified structure

    Args:
    fp_2solutes = path to folder containing GB with 2 solutes
    """
    structure = Structure.from_file("%s\\CONTCAR" % (fp_2solutes))
    # get pair of non-Fe site idx
    distance_pair = [
        i for i, site in enumerate(structure) if site.species_string != "Fe"
    ]
    distance = structure[distance_pair[1]].distance(structure[distance_pair[0]])

    return distance


def get_totalenergy(GB, structure_name="GB"):
    """
    Returns a total energy (eV) value for a specified GB (1 sol case)

    Args:
    path = path to directory containing structure
    """
    if structure_name == "GB":
        df = pd.read_csv("%s\\%s\\Co\\info.csv" % (fp_Seg1_path, os.path.basename(GB)))
        E_GB = df[df["system"] == "GB"].energy.values[0]
    else:
        structure_name = "-".join(structure_name.split(sep="-")[0:-2])
        element = structure_name.split(sep="-")[0]
        df = pd.read_csv(
            "%s\\%s\\%s\\info.csv" % (fp_Seg1_path, os.path.basename(GB), element)
        )
        E_GB = df[df["system"] == structure_name].energy.values[0]

    return E_GB


def get_1sol_df(folder_path, midpoint=0.5000):
    """
    Returns a pandas dataframe containing segregation energy, voronoi volume, and magnetic moment profiles
    for all 1 solute cases in a specified GB folder: "folder_path"
    """
    results = pd.read_csv("%s\\info.csv" % folder_path)
    # Energy of the pure slab structure
    E_slab = results.loc[results["system"] == "SLAB"]["energy"].values[0]
    # Energy of the slab + 1 solute structure
    E_slab_imp = results.loc[results["system"].str.contains("-SLAB-")]["energy"].values[
        0
    ]
    # Energy of the pure GB
    E_GB = results.loc[results["system"] == "GB"]["energy"].values[0]
    # Drop any results related to pure GB and any slab structures
    df = results.copy().loc[~results["system"].str.contains("SLAB|GB")]
    # Calculate energy of segregation
    df["E_seg"] = df["energy"] - E_GB - (E_slab_imp - E_slab)
    ## This section assigns a distance from GB for the solute
    distance_compiled = []
    magmom_list_compiled = []
    magmom_compiled = []
    vvol_list_compiled = []
    vvol_compiled = []
    for _, system in df.iterrows():
        # Read the CONTCAR structure in the folder
        structure = Structure.from_file(
            "%s\\%s\\CONTCAR" % (folder_path, system["system"])
        )
        # Get solute number in structure
        solute_no = int(system["system"].split(sep="-")[1:2][0])
        solute_no = [
            i for i, site in enumerate(structure) if site.species_string != "Fe"
        ][0]
        # Distance to GB plane (center plane frac z)
        distance = (
            abs(midpoint - structure[solute_no].frac_coords[-1]) * structure.lattice.c
        )
        distance_compiled.append(distance)
        # Magnetic moment
        magmom_df = pd.read_csv(
            "%s\\%s\\magnet.csv" % (folder_path, system["system"]),
            delim_whitespace=True,
            header=None,
        )[[1, 2, 3, 4]]
        ## magnetic moment list
        magmom_list = list(magmom_df[4].values)
        magmom_list_compiled.append(magmom_list)
        ## magnetic moment of non Fe solute
        magmom = magmom_list[solute_no]
        magmom_compiled.append(magmom)
        # Voronoi volume
        vvol_df = pd.read_excel("%s\\%s\\vvol.xlsx" % (folder_path, system["system"]))
        ## Voronoi volume list
        vvol_list = json.loads(vvol_df.iloc[1].values[1])
        vvol_list_compiled.append(vvol_list)
        ## Voronoi volume of non Fe solute
        vvol = vvol_list[solute_no]
        vvol_compiled.append(vvol)
    df["distance_GB"] = distance_compiled
    df["magmom"] = magmom_compiled
    df["magmom_list"] = magmom_list_compiled
    df["vvol"] = vvol_compiled
    df["vvol_list"] = vvol_list_compiled
    df["d"] = [row.system.split(sep="-")[-1] for i, row in df.iterrows()]
    df["site"] = [row.system.split(sep="-")[1] for i, row in df.iterrows()]

    return df


def get_area(path):
    structure = Structure.from_file(path)
    area = structure.volume / structure.lattice.c
    return area


def get_1sol_etarel_cohesion_df(folder_path):
    """
    Returns Wsep_rel df for all elements in a GB folder

    Inputs:
    GB_path = directory path to GB containing cleaved relaxed surfaces
    """

    def getEnergyFromData_1sol(GB_name, element, site):

        # returns total energy of specific site and permutation
        if element == "GB":
            data_specified_path = "%s\\%s\\%s\\info.csv" % (fp_Seg1_path, GB_name, "Co")
        elif GB_name == "S9-RA110-S2-21" and element == "P":
            # This is a special exception in the case of the Sigma 9 (2-21) GB, where I calculated 2x1 GB.
            # To fix the segregation profile.
            # Obviously, you can't compute a work of separation comparing 2x1 GB to 1x1 energies.
            # So, I am adding this exception.
            data_specified_path = "%s\\%s\\%s\\info-old.csv" % (
                fp_Seg1_path,
                GB_name,
                element,
            )
        else:
            data_specified_path = "%s\\%s\\%s\\info.csv" % (
                fp_Seg1_path,
                GB_name,
                element,
            )
        df = pd.read_csv(data_specified_path)
        df["solute_no"] = [
            (
                int(row.system.split(sep="-")[1])
                if not any(slabgb in row.system for slabgb in ["SLAB", "GB"])
                else "NaN"
            )
            for _, row in df.iterrows()
        ]
        if element == "GB":
            # listen here you little shit
            energy = df.loc[df["system"] == "GB"].energy.values[0]
        else:
            energy = df.loc[df["solute_no"] == site].energy.values[0]

        return energy

    df = pd.read_csv("%s\\info.csv" % folder_path)
    df["area"] = [
        get_area("%s\\%s\\CONTCAR" % (folder_path, cleave_case.system))
        for _, cleave_case in df.iterrows()
    ]
    df["element"] = [row.system.split(sep="-")[0] for _, row in df.iterrows()]
    df["site"] = [
        int(row.system.split(sep="-")[1]) if "GB" not in row.system else "GB"
        for _, row in df.iterrows()
    ]
    df["GB_energy"] = [
        getEnergyFromData_1sol(os.path.basename(folder_path), row.element, row.site)
        for _, row in df.iterrows()
    ]
    df["Wsep_rel"] = np.round(
        (df["energy"] - df["GB_energy"]) * 16.02176565 / (df["area"]), 3
    )
    df["system_base"] = [
        "-".join(row.system.split(sep="-")[0:-2]) for _, row in df.iterrows()
    ]

    return df


def get_1sol_etarigid_cohesion_df(folder_path):
    """
    Returns Wsep_RGS values for all elements in a GB folder

    Inputs:
    GB_path = directory path to GB containing RGS surfaces
    """
    case_df_list = []
    for cases in get_immediate_subdirectories(folder_path):
        # print(cases)
        results = pd.read_csv("%s\\info.csv" % cases)
        GB_energy = results[results["system"] == os.path.basename(cases)][
            "energy"
        ].values[0]
        results["area"] = [
            get_area("%s\\%s\\CONTCAR" % (cases, cleave_case.system))
            for _, cleave_case in results.iterrows()
        ]
        results["Wsep"] = (
            (results["energy"] - GB_energy) * 16.02176565 / (results["area"])
        )
        # print(results.loc[0::, ['system','Wsep']])
        df = results[results["Wsep"] > 0.0001]
        new_df = df.copy()
        new_df["cleavage_plane_name"] = [
            x.split(sep="-")[-2] for x in df["system"].values
        ]
        new_df["cleavage_plane"] = [
            float(x.split(sep="-")[-1]) for x in df["system"].values
        ]
        # Create a single row DF with column names assigned cleavage plane values
        case_df = pd.DataFrame(new_df["Wsep"].values).transpose()
        case_df.columns = new_df["cleavage_plane_name"].values
        case_df["cleavage_planes"] = [new_df["cleavage_plane"].values]
        case_df["cp_names"] = [new_df["cleavage_plane_name"].values]
        case_df["system"] = os.path.basename(cases)
        case_df["Wsep_RGS_list"] = [np.round(new_df["Wsep"].values, 3)]
        case_df_list.append(case_df)
    GB_df = np.round(pd.concat(case_df_list), 2)
    GB_df["Wsep_RGS"] = [
        np.round(min(wsep_lists), 3) for wsep_lists in GB_df["Wsep_RGS_list"]
    ]
    GB_df["min_cp_name"] = [
        row.cp_names[np.argmin(row["Wsep_RGS_list"])] for _, row in GB_df.iterrows()
    ]
    GB_df["min_cp"] = [
        row.cleavage_planes[np.argmin(row["Wsep_RGS_list"])]
        for _, row in GB_df.iterrows()
    ]
    return GB_df


def cp_bondorder(
    structure_path="%s\\CONTCAR" % os.getcwd(),
    DDEC_output_path="%s" % os.getcwd(),
    cleavage_plane_array=[0.5],
    bo_threshold=0.0,
):
    # This function calculates the bond order sum and returns a value, given a structure and chargemol output path
    # Read the DDEC Output and convert it into a csv temp file
    structure = Structure.from_file(structure_path)
    VASPDDEC_2_CSV(
        "%s\\VASP_DDEC_analysis.output" % DDEC_output_path,
        "%s\\chargemol.csv" % os.getcwd(),
    )
    chargemol_data = pd.read_csv(
        "%s\\chargemol.csv" % os.getcwd(), delim_whitespace=True
    )
    chargemol_data = chargemol_data[chargemol_data["final_bond_order"] > bo_threshold]
    bond_data = chargemol_data.copy()[
        ["atom1", "atom2", "repeata", "repeatb", "final_bond_order"]
    ]
    # -1 because chargemol begins indexing at 1, equivalent to structure[0]
    bond_data["atom1pos"] = [
        structure[x - 1].frac_coords for x in bond_data["atom1"].values
    ]
    bond_data["atom2pos"] = [
        structure[x - 1].frac_coords for x in bond_data["atom2"].values
    ]
    # zpos fractional
    bond_data["atom1zpos"] = [
        structure[x - 1].frac_coords[-1] for x in bond_data["atom1"].values
    ]
    bond_data["atom2zpos"] = [
        structure[x - 1].frac_coords[-1] for x in bond_data["atom2"].values
    ]
    # zpos fractional
    bond_data["atom1_ele"] = [
        structure[x - 1].species_string for x in bond_data["atom1"].values
    ]
    bond_data["atom2_ele"] = [
        structure[x - 1].species_string for x in bond_data["atom2"].values
    ]
    cp_bo = []
    clp_df_list = []
    for cp in cleavage_plane_array:
        # cleavage plane and find the sum of bond orders passing through it
        clp_df = bond_data[
            (bond_data[["atom2zpos", "atom1zpos"]].max(axis=1) > cp)
            & (bond_data[["atom2zpos", "atom1zpos"]].min(axis=1) < cp)
        ]
        clp_df = clp_df.copy()[(clp_df["repeata"] == 0) | (clp_df["repeatb"] == 0)]
        # We only want to calculate for atoms that exist  cell. This is important for bond order/area normalisation
        clp_df_countonce = clp_df.copy()[
            (clp_df["repeata"] == 0) & (clp_df["repeatb"] == 0)
        ]
        clp_df_counthalf = clp_df.copy()[
            (clp_df["repeata"] != 0) | (clp_df["repeatb"] != 0)
        ]
        # Basic summed bond order over CP
        final_bond_order = (
            clp_df_countonce.final_bond_order.sum()
            + 0.5 * clp_df_counthalf.final_bond_order.sum()
        )
        # N largest
        # final_bond_order = clp_df.nlargest(15, ['final_bond_order'])["final_bond_order"].sum()
        # IMPORTANT: This assumes that the cross sectional area can be calculated this way
        a_fbo = final_bond_order / (
            float(structure.lattice.volume) / float(structure.lattice.c)
        )
        # print("area of this is %s" % (float(structure.lattice.volume)/float(structure.lattice.c)))
        cp_bo.append(a_fbo)
        clp_df_list.append(clp_df)
    return cp_bo, clp_df_list


def VASPDDEC_2_CSV(filename, output_filename):
    flist = open(filename).readlines()
    parsing = False
    matrix = []
    for line in flist:
        if "The legend for the bond pair matrix follows:" in line:
            parsing = False
        if parsing:
            matrix.append(line)
            # print(line)
        if "The final bond pair matrix is" in line:
            parsing = True
    f = open(output_filename, "w")
    f.write(
        "atom1 atom2 repeata repeatb repeatc "
        + "min-na max-na min-nb max-nb min-nc max-nc contact-exchange avg-spin-pol-bonding-term overlap-population "
        + "isoaepfcbo coord-term-tanh pairwise-term exp-term-comb-coord-pairwise "
        + "bond-idx-before-self-exch final_bond_order \n"
    )
    for bond in matrix:
        f.write(bond)
    f.close()


def get_1sol_site_SBO(GB_path):
    """
    Returns summed bond order (DDEC6) for a single-solute case
    and an array containing the atoms it has bonds with in an array

    Inputs:
    GB_path = directory path to GB
    """
    structure = Structure.from_file("%s\\CONTCAR" % GB_path)
    solute_no = [i for i, site in enumerate(structure) if site.species_string != "Fe"][
        0
    ]
    BO_dict = get_BondOrderInfo(GB_path)
    SBO = BO_dict[solute_no]["bond_order_sum"]
    atoms_bond_array = []
    for i in BO_dict[solute_no]["bonded_to"]:
        atoms_bond_array.append(i["index"])
    return SBO, atoms_bond_array, solute_no


def get_BondOrderInfo(filename):
    """
    Internal command to process pairwise bond order information
    Args:
        filename (str): The path to the DDEC6_even_tempered_bond_orders.xyz file
    """
    filename = filename + "\\DDEC6_even_tempered_bond_orders.xyz"
    # Get where relevant info for each atom starts
    bond_order_info = {}

    with open(filename) as r:
        for line in r:
            l = line.strip().split()
            if "Printing BOs" in line:
                start_idx = int(l[5]) - 1
                start_el = Element(l[7])
                bond_order_info[start_idx] = {"element": start_el, "bonded_to": []}
            elif "Bonded to the" in line:
                direction = tuple(int(i.split(")")[0].split(",")[0]) for i in l[4:7])
                end_idx = int(l[12]) - 1
                end_el = Element(l[14])
                bo = float(l[20])
                spin_bo = float(l[-1])
                bond_order_info[start_idx]["bonded_to"].append(
                    {
                        "index": end_idx,
                        "element": end_el,
                        "bond_order": bo,
                        "direction": direction,
                        "spin_polarization": spin_bo,
                    }
                )
            elif "The sum of bond orders for this atom" in line:
                bond_order_info[start_idx]["bond_order_sum"] = float(l[-1])

    return bond_order_info


def get_site_SBO(filename, site):
    """
    Internal command to process pairwise bond order information
    Args:
        filename (str): The path to folder containing the DDEC6_even_tempered_bond_orders.xyz file
        site (int): Do not
    """
    # Get where relevant info for each atom starts
    bond_order_info = {}
    filename = filename + "\\DDEC6_even_tempered_bond_orders.xyz"
    with open(filename) as r:
        for line in r:
            l = line.strip().split()
            if "Printing BOs" in line:
                start_idx = int(l[5]) - 1
                start_el = Element(l[7])
                bond_order_info[start_idx] = {"element": start_el, "bonded_to": []}
            elif "Bonded to the" in line:
                direction = tuple(int(i.split(")")[0].split(",")[0]) for i in l[4:7])
                end_idx = int(l[12]) - 1
                end_el = Element(l[14])
                bo = float(l[20])
                spin_bo = float(l[-1])
                bond_order_info[start_idx]["bonded_to"].append(
                    {
                        "index": end_idx,
                        "element": end_el,
                        "bond_order": bo,
                        "direction": direction,
                        "spin_polarization": spin_bo,
                    }
                )
            elif "The sum of bond orders for this atom" in line:
                bond_order_info[start_idx]["bond_order_sum"] = float(l[-1])
    #     site_sbo = 0
    #     for j in bond_order_info[site]['bonded_to']:
    #         site_sbo += j['bond_order']
    site_sbo = bond_order_info[site]["bond_order_sum"]
    return site_sbo


def get_solution_energy(GB, element):
    """
    Returns solution energy for an element of a single-solute case in eV

    Inputs:
    GB_path = directory path to GB
    element = element string, e.g. for chromium it would be "Cr"
    """
    results = pd.read_csv("%s\\%s\\%s\\info.csv" % (fp_Seg1_path, GB, element))
    # Energy of the pure slab structure
    E_slab = results.loc[results["system"] == "SLAB"]["energy"].values[0]
    # Energy of the slab + 1 solute structure
    E_slab_imp = results.loc[results["system"].str.contains("-SLAB-")]["energy"].values[
        0
    ]
    # Energy of solution
    solution_energy = E_slab_imp - E_slab

    return solution_energy


def get_1sol_cohesion_summary(GB_string):
    """
    Returns a cohesion-energy of segregation summary df that is used for generating
    single solute cohesion effect vs segregation tendency plots
    """
    RGS_1sol_df = get_1sol_etarigid_cohesion_df("%s\\%s" % (fp_Wsep1_rigid, GB_string))
    Wsep_rel_1sol_df = get_1sol_etarel_cohesion_df("%s\\%s" % (fp_Wsep1_rel, GB_string))
    rig_df_merge = RGS_1sol_df.copy()[
        ["system", "Wsep_RGS", "cleavage_planes", "Wsep_RGS_list"]
    ]
    rel_df_merge = Wsep_rel_1sol_df.copy()[["system_base", "Wsep_rel"]]
    rel_df_merge = rel_df_merge.rename(columns={"system_base": "system"})
    df = pd.merge(rig_df_merge, rel_df_merge)
    df["eta_RGS"] = [
        np.round(row.Wsep_RGS - df.loc[df["system"] == "GB"].Wsep_RGS.values[0], 2)
        for _, row in df.iterrows()
    ]
    df["eta_rel"] = [
        np.round(row.Wsep_rel - df.loc[df["system"] == "GB"].Wsep_rel.values[0], 2)
        for _, row in df.iterrows()
    ]
    df["d_eta"] = df["eta_rel"] - df["eta_RGS"]
    df["eta_rel_pct"] = (df["eta_rel"] * 100) / df.loc[
        df["system"] == "GB"
    ].Wsep_rel.values[0]
    df["eta_RGS_pct"] = (df["eta_RGS"] * 100) / df.loc[
        df["system"] == "GB"
    ].Wsep_RGS.values[0]
    df["element"] = [x.system.split(sep="-")[0] for _, x in df.iterrows()]

    min_bo = []
    bo_df_list = []
    bo_array_list = []
    for idx, row in RGS_1sol_df.iterrows():
        cp_array = row["cleavage_planes"]
        system = row["system"]
        bo_array, bodf_list = cp_bondorder(
            structure_path="%s\\%s\\%s\\CONTCAR" % (fp_BO1, GB_string, system),
            DDEC_output_path="%s\\%s\\%s" % (fp_BO1, GB_string, system),
            cleavage_plane_array=cp_array,
            bo_threshold=0,
        )
        bo_array_list.append(bo_array)
        min_bo.append(min(bo_array))

        bo_df = bodf_list[np.argmin(bo_array)][
            ["atom1_ele", "atom2_ele", "atom1", "atom2", "final_bond_order"]
        ]
        bo_df = bo_df[bo_df["final_bond_order"] > 0.01]
        bo_df_list.append(bo_df)

    df["ANSBO"] = min_bo
    df["eta_ANSBO"] = [
        row.ANSBO - df.loc[df["system"] == "GB"].ANSBO.values[0]
        for _, row in df.iterrows()
    ]
    df["bond_df"] = bo_df_list
    eseg_list = []
    # Section that extracts information about summed bond orders
    SBO_list = []
    pGB_SBO_list = []
    for i, row in df.iterrows():
        if row.system == "GB":
            SBO = np.nan
            pGB_SBO = np.nan
        else:
            SBO = get_1sol_site_SBO("%s\\%s\\%s" % (fp_BO1, GB_string, row.system))[0]
            solute_no = get_1sol_site_SBO(
                "%s\\%s\\%s" % (fp_BO1, GB_string, row.system)
            )[2]
            pGB_SBO = get_site_SBO("%s\\%s\\%s" % (fp_BO1, GB_string, "GB"), solute_no)
            # print("%s\\%s\\%s" % (fp_BO1, GB_string, row.system), solute_no)
        SBO_list.append(SBO)
        pGB_SBO_list.append(pGB_SBO)
    df["site_SBO"] = SBO_list
    df["site_pGB_SBO"] = pGB_SBO_list
    df["site_SBO_delta"] = df["site_SBO"] - df["site_pGB_SBO"]
    # Exception in the case of the S11-RA110-S3-32 case:
    # Take the segregation energies from the 2x2 cell instead of the 1x1 cell
    # This was done since interface reconstruction occurs heavily in the 1x1 cell
    if GB_string == "S11-RA110-S3-32":
        GB_string = "S11-RA110-S3-32-2x2"
    for _, row in df.iterrows():
        if row.system.split(sep="-")[0] != "GB":
            # print(row.system.split(sep="-")[0])
            eseg_df = get_1sol_df(
                "%s\\%s\\%s" % (fp_Seg1_path, GB_string, row.system.split(sep="-")[0]),
                midpoint=0.5094,
            )
            eseg = np.round(
                eseg_df.loc[eseg_df["system"] == row.system].E_seg.values[0], 3
            )
        else:
            eseg = np.nan
        eseg_list.append(eseg)
    df["E_seg"] = eseg_list

    return df
