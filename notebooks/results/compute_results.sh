#!/bin/bash

# *** INSTLL LIBRAIRIES OF results.ipynb BEFORE RUNNING THIS SCRIPT ***

# loop over dirs in /storage/smiles2spec_models/
dirs=()

for dir in /storage/smiles2spec_models/*/; do
    # if [[ -d "$dir" && ! $(basename "$dir") =~ ^old_ ]]; then
    # if [[ -d "$dir" && ! $(basename "$dir") =~ ^smiles_comp_ ]]; then
    if [[ -d "$dir" && $(basename "$dir") =~ ^smiles_comp_exp ]]; then
        dirs+=("$dir")  # Add directory to the array
    fi
done

# Print the list of directories
echo
echo "Directories stored in the array:"
printf "%s\n" "${dirs[@]}"
echo

# Loop
for f in "${dirs[@]}"; do

    echo "Computing results: $f"

    TRANSFORMERS_NO_PROGRESS_BAR=true python compute_results.py \
        --results_folder "$f"\
        --data_folder "/storage/smiles2spec_data" \
        --new_results_folder "/notebooks/smiles2spec/notebooks/results"

    echo "Results computed."
    echo

done
