import argparse
from utils.generic import find_and_compress_directories_parallel
import os

def main():
    parser = argparse.ArgumentParser(description='Find and compress directories based on specified criteria.')
    parser.add_argument('directory', metavar='DIR', type=str, help='the directory to operate on')
    args = parser.parse_args()

    find_and_compress_directories_parallel(
        parent_dir=args.directory,
        valid_dir_if_filenames=["INCAR", "POTCAR"],
        exclude_files_from_tarball=["CHG", "CHGCAR"],
        exclude_filepatterns_from_tarball=["AECCAR*"],
        keep_after=True,
        files=[],
        file_patterns=[],
        print_msg=True,
        inside_dir=True
    )

if __name__ == '__main__':
    main()
