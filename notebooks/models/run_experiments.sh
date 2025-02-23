#!/bin/bash

# *** INSTLL LIBRAIRIES OF LLM_Base_Model.ipynb BEFORE RUNNING THIS SCRIPT ***

export TQDM_DISABLE=1

# "DeepChem/ChemBERTa-5M-MTR" "ncfrey/ChemGPT-4.7M"
models=("DeepChem/ChemBERTa-5M-MTR") # "ncfrey/ChemGPT-4.7M" "DeepChem/ChemBERTa-5M-MTR"

data=("exp" "comp_exp")            # "comp" "exp" "comp_exp" do comp and exp separately?
nb_epochs=16             # 15 originally; 16 if finetuning mode
hidden_layers=(0 1 3)    # (0 1 3 5)

# Loop
for m in "${models[@]}"; do
    for d in "${data[@]}"; do 
        for h in "${hidden_layers[@]}"; do

            echo "Running experiment with model=$m, hidden_layers=$h, and epochs=$nb_epochs"

            TRANSFORMERS_NO_PROGRESS_BAR=true 
            python train_model.py \
                --inputs smiles \
                --data_type "$data" \
                --model "$m" \
                --hidden_layers "$h" \
                --hidden_dim 2200 \
                --loss "MSE" \
                --epochs "$nb_epochs" >> "experiments_${m#*/}_${data#*/}.log" 2>&1

            echo "Experiment completed."
            echo
            
        done      
    done  
done
