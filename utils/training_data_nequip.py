import os
from ase.io import read, write
import numpy as np
import pandas as pd
import time


def process_list(my_list, n):
    # if list has length 1 or 2, return the list as is
    if len(my_list) <= 2:
        return my_list

    # get the first and last elements of the list
    first = my_list[0]
    last = my_list[-1]

    # get every nth element of the list, excluding first and lasat images
    new_list = [
        my_list[i]
        for i in range(0, len(my_list), n)
        if i != 0 and i != len(my_list) - 1
    ]

    # return the result
    return [first] + new_list + [last]


def extract_allegro_extxyz(
    filepath,
    max_electronic_steps=120,
    every_nth_image=4,
    scf_steps=[],
    output_filepath="allegro_training_data.extxyz",
):
    filtered_list = []
    print(filepath)
    ase_outcar = read(filepath, format="vasp-out", index=":")
    if scf_steps:
        for j, n_electronic_steps in enumerate(scf_steps):
            if n_electronic_steps != max_electronic_steps:
                filtered_list.append(ase_outcar[j])
    else:
        filtered_list = ase_outcar

    every_n_list = process_list(filtered_list, every_nth_image)
    for _, atoms_obj in enumerate(every_n_list):
        write(output_filepath, atoms_obj, append=True, format="extxyz")


import glob

df_pickles_filelist = []
for file in glob.glob("S*.pkl"):
    if file != "S11-RA110-S3-32.pkl":
        continue
    print(file)
    df_pickles_filelist.append(file)

import multiprocessing

start_time = time.time()

num_processors = multiprocessing.cpu_count()
if len(df_pickles_filelist) < num_processors:
    processes = len(df_pickles_filelist)
else:
    processes = num_processors
print(f"Number of processors: {num_processors}, used: {processes}")


def allegro_data_setup_from_df(df_pickle_filepath):
    df = pd.read_pickle(df_pickle_filepath)
    for _, row in df.iterrows():
        output_file = os.path.basename(df_pickle_filepath).split(sep=".pkl")[0]
        extract_allegro_extxyz(
            row.filepath,
            scf_steps=row.scf_steps,
            output_filepath=f"{output_file}-AllegroNequip.extxyz",
        )


with multiprocessing.Pool(processes=processes) as pool:
    pool.map(allegro_data_setup_from_df, df_pickles_filelist)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", np.round(elapsed_time, 3), "seconds")
