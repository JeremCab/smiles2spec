#!/bin/bash

model="DeepChem/ChemBERTa-5M-MTR"
data="comp"
nb_epochs=5

hidden_layers=(0 1 3 5)

# Loop
for h in "${hidden_layers[@]}"; do

    echo "Running experiment with model=$model, hidden_layers=$h, and epochs=$nb_epochs"
    
    RANSFORMERS_NO_PROGRESS_BAR=true python3 train_model.py \
        --inputs smiles \
        --data_type "$data" \
        --model "$model" \
        --hidden_layers "$h" \
        --input_dim 200 \
        --hidden_dim 2200 \
        --epochs "$nb_epochs" >> experiments.log 2>&1
    
    echo "Experiment completed."
    echo
    
done
