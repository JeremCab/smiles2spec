#!/bin/bash

export TQDM_DISABLE=1

# "DeepChem/ChemBERTa-5M-MTR" "ncfrey/ChemGPT-4.7M"
models=("DeepChem/ChemBERTa-5M-MTR" "ncfrey/ChemGPT-4.7M") 

data="exp"
nb_epochs=5

hidden_layers=(0 1 3 5) # (0 1 3 5)

# Loop
for m in "${models[@]}"; do
    for h in "${hidden_layers[@]}"; do

        echo "Running experiment with model=$m, hidden_layers=$h, and epochs=$nb_epochs"

        TRANSFORMERS_NO_PROGRESS_BAR=true python3 train_model.py \
            --inputs smiles \
            --data_type "$data" \
            --model "$model" \
            --hidden_layers "$h" \
            --input_dim 200 \
            --hidden_dim 2200 \
            --epochs "$nb_epochs" >> experiments_ChemBERTa_exp.log 2>&1

        echo "Experiment completed."
        echo

    done
done
