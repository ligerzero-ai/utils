import os
import argparse
import pandas as pd

def analyze_vasp_database(folder_path, output_compression=False):
    # Initialize paths for both potential database files
    database_file_pkl = os.path.join(folder_path, 'vasp_database.pkl')
    database_file_gz = os.path.join(folder_path, 'vasp_database.pkl.gz')

    # Determine which file exists and set the appropriate path and compression option
    if os.path.exists(database_file_gz):
        database_file = database_file_gz
        compression_option = 'gzip'
    elif os.path.exists(database_file_pkl):
        database_file = database_file_pkl
        compression_option = None
    else:
        print("Error: neither 'vasp_database.pkl' nor 'vasp_database.pkl.gz' found in the specified folder.")
        return

    # Load the database into a DataFrame with or without compression
    df = pd.read_pickle(database_file, compression=compression_option)

    # Print and sort the relevant columns
    print(df[["job_name", "convergence", "filepath"]].sort_values(by="job_name"))

    # Calculate the number of failed and converged jobs
    failed_jobs = df[df["convergence"] == False]
    converged_jobs = df[df["convergence"] == True]

    # Determine compression option for output based on the user input
    output_compression_option = 'gzip' if output_compression else None

    # Write the failed_jobs and converged_jobs DataFrames to separate pickle files with optional compression
    failed_jobs.to_pickle(os.path.join(folder_path, 'failed_jobs.pkl.gz' if output_compression else 'failed_jobs.pkl'), compression=output_compression_option)
    converged_jobs.to_pickle(os.path.join(folder_path, 'converged_jobs.pkl.gz' if output_compression else 'converged_jobs.pkl'), compression=output_compression_option)

    # Print the counts
    print(f"The number of failed jobs is: {len(failed_jobs)}")
    print(f"The number of successful jobs is: {len(converged_jobs)}")
    print(f"The total number of jobs is: {len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze VASP database")
    parser.add_argument("folder_path", type=str, help="Folder path containing 'vasp_database.pkl' or 'vasp_database.pkl.gz'")
    parser.add_argument("--output_compression", action="store_true", help="Enable gzip compression for output pkl files")
    args = parser.parse_args()
    analyze_vasp_database(args.folder_path, args.output_compression)