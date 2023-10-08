import os
import argparse
import pandas as pd

def analyze_vasp_database(folder_path):
    # Check if the folder contains the 'vasp_database.pkl' file
    database_file = os.path.join(folder_path, 'vasp_database.pkl')

    if not os.path.exists(database_file):
        print("Error: 'vasp_database.pkl' not found in the specified folder.")
        return

    # Load the database into a DataFrame
    df = pd.read_pickle(database_file)

    # Print and sort the relevant columns
    print(df[["job_name", "convergence", "filepath"]].sort_values(by="job_name"))

    # Calculate the number of failed and converged jobs
    failed_jobs = df[df["convergence"] == False]
    converged_jobs = df[df["convergence"] == True]

    # Write the failed_jobs and converged_jobs DataFrames to separate pickle files
    failed_jobs.to_pickle(os.path.join(folder_path, 'failed_jobs.pkl'))
    converged_jobs.to_pickle(os.path.join(folder_path, 'converged_jobs.pkl'))

    # Print the counts
    print(f"The number of failed jobs is: {len(failed_jobs)}")
    print(f"The number of successful jobs is: {len(converged_jobs)}")
    print(f"The total number of jobs is: {len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze VASP database")
    parser.add_argument("folder_path", type=str, help="Folder path containing 'vasp_database.pkl'")
    args = parser.parse_args()
    analyze_vasp_database(args.folder_path)