from utils.vasp.vasp import DatabaseGenerator
import argparse
import warnings
from multiprocessing import cpu_count


def main():
    warnings.filterwarnings("ignore")

    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Find and compress directories based on specified criteria."
    )
    parser.add_argument(
        "directory", metavar="DIR", type=str, help="the directory to operate on"
    )
    parser.add_argument(
        "--max_dir_count",
        type=int,
        help="Maximum directory count for database generation",
    )
    parser.add_argument(
        "--read_all_runs_in_dir",
        action="store_true",
        default=False,
        help="Read all runs in directory",
    )
    parser.add_argument(
        "--read_error_runs_in_dir",
        action="store_true",
        default=False,
        help="Read directories with errors",
    )
    args = parser.parse_args()

    datagen = DatabaseGenerator(args.directory, max_workers=cpu_count())

    # Check if max_dir_count is provided as an argument
    if args.max_dir_count is not None:
        max_dir_count = args.max_dir_count
    else:
        max_dir_count = 2000  # Default value

    # Call the update_failed_jobs_in_database function with the updated parameters
    df = datagen.update_failed_jobs_in_database(
        df_path=args.directory,
        read_error_dirs=args.read_error_runs_in_dir,
        read_multiple_runs_in_dir=args.read_all_runs_in_dir,
        max_dir_count=max_dir_count,
        df_compression=True,
    )


if __name__ == "__main__":
    main()
