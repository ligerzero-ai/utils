#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi

folder_path="$1"

# Setonix
path_to_utils="/scratch/pawsey0380/hmai/utils/actual_usage/"
# Garching
path_to_utils="/cmmc/u/hmai/personal_dev/utils/actual_usage"

# Run the first Python script and append the output to py.output
python "$path_to_utils/build_vasp_database.py" "$folder_path"

# Run the second Python script and append the output to py.output
python "$path_to_utils/summarise_vasp_database.py" "$folder_path"