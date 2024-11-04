#!/bin/bash

# export TQDM_DISABLE=1 # XXX

# "DeepChem/ChemBERTa-5M-MTR" "ncfrey/ChemGPT-4.7M"
models=("DeepChem/ChemBERTa-5M-MTR") # "ncfrey/ChemGPT-4.7M" "DeepChem/ChemBERTa-5M-MTR"

data="comp"  # exp remains XXX
nb_epochs=10 # 5 epochs are too few

hidden_layers=(3) # (0 1 3 5)

# Loop
for m in "${models[@]}"; do
    for h in "${hidden_layers[@]}"; do

        echo "Running experiment with model=$m, hidden_layers=$h, and epochs=$nb_epochs"
        
        # XXX
        # TRANSFORMERS_NO_PROGRESS_BAR=true 
        python train_model.py \
            --inputs smiles \
            --data_type "$data" \
            --model "$m" \
            --hidden_layers "$h" \
            --input_dim 200 \
            --hidden_dim 2200 \
            --epochs "$nb_epochs" # >> "experiments_${m#*/}_${data#*/}.log" 2>&1 # XXX

        echo "Experiment completed."
        echo

    done
done
