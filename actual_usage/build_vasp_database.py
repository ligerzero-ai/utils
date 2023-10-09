from utils.vasp import DatabaseGenerator
import argparse

def main():
    parser = argparse.ArgumentParser(description='Find and compress directories based on specified criteria.')
    parser.add_argument('directory', metavar='DIR', type=str, help='the directory to operate on')
    parser.add_argument('--extract', action='store_true', help='Extract directories during database generation')
    parser.add_argument('--max_dir_count', type=int, help='Maximum directory count for database generation')
    args = parser.parse_args()

    datagen = DatabaseGenerator(args.directory)
    
    # Check if max_dir_count is provided as an argument
    if args.max_dir_count is not None:
        max_dir_count = args.max_dir_count
    else:
        max_dir_count = 2000  # Default value
    
    df = datagen.build_database(max_dir_count=max_dir_count, extract_directories=args.extract)

if __name__ == '__main__':
    main()
