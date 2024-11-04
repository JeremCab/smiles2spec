#!/usr/bin/env python
# coding: utf-8


# ================== #
# Fine-Tuning Models #
# ================== #



# --------- #
# Libraries #
# --------- #

import logging
import os
import shutil
import time
import argparse
import pickle
import wandb

import torch
import torch.nn as nn

import transformers
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput

import datasets
from datasets import load_from_disk
# from datasets import load_metric

from huggingface_hub import login

# supress progress bars (for cleaner log files)
# logging.basicConfig(level=logging.ERROR)
# os.environ["TQDM_DISABLE"] = "1"


# --------- #
# Arguments #
# --------- #

# example of command line:
# python3 train_model.py --inputs smiles --data_type comp --model DeepChem/ChemBERTa-5M-MTR --hidden_layers 0 --input_dim 200 --hidden_dim 2200 --epochs 0.1 >> experiment.log 2>&1

# Create argument parser
parser = argparse.ArgumentParser(description="Training arguments...")

# 1.
parser.add_argument("--inputs", choices=["smiles", "selfies"], default="smiles", help="SMILES or SELFIES")

# 2.
parser.add_argument("--data_type", choices=["comp", "exp"], default="comp", help="Computational or experimental spectra")

# 3.
parser.add_argument("--model",  type=str, default="DeepChem/ChemBERTa-5M-MTR", help="Computational or experimental spectra")

# 4.
parser.add_argument("--hidden_layers", type=int, default=0, help="Number of hidden layers for the FFNN")

# 5.
parser.add_argument("--input_dim", type=int, default=200, help="Input dimension of the FFNN")

# 6.
parser.add_argument("--hidden_dim", type=int, default=2200, help="Hidden dimension of the FFNN")

# 7. *** from here, the parameters are usually the default ones... ***
parser.add_argument("--epochs", type=float, default=5, help="Number of epochs")

# 8.
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

# 9.
parser.add_argument("--finetuning", type=bool, default=False, help="Whether too finetune on experimental data")

# Parse arguments
args = parser.parse_args()


# -------- #
# Preamble #
# -------- #


# 1. Choose your input mode: "smiles" or "selfies" whether to predict from SMILES or SELFIES
INPUTS = args.inputs

# 2. Choose training on computed or experimental spectra
DATA_TYPE = args.data_type

# 3. Choose model and its parameters:
#    "DeepChem/ChemBERTa-5M-MTR"
#    "ncfrey/ChemGPT-4.7M"
#    ...
MODEL_NAME = args.model
if MODEL_NAME == "ncfrey/ChemGPT-4.7M":
    INPUTS = "selfies"
    
MODEL_SUFFIX = MODEL_NAME.split("/")[1]

print("*** Start experiment ***")
print(f"Inputs:    {INPUTS}")
print(f"Data type: {DATA_TYPE}")
print(f"Model:     {MODEL_NAME}")

MODEL_CACHE = "/storage/smiles2spec_models"
SPECIFICATIONS = f"{INPUTS}_{DATA_TYPE}_{MODEL_SUFFIX}_FFNN-{args.hidden_layers}-{args.input_dim}-{args.hidden_dim}"

RESULTS_FOLDER = os.path.join(MODEL_CACHE, SPECIFICATIONS)
print(f"Results folder: {RESULTS_FOLDER}")


# ---------- #
# Parameters #
# ---------- #


# Model parameters (populated from arguments)

args_d = {
    'model_name': MODEL_NAME,
    'output_activation': 'exp',
    'norm_range': None, # (50, 550),
    'dropout': 0.2,
    'activation': nn.ReLU(),
    'ffn_num_layers': args.hidden_layers, # e.g., 0, 1, 3, 5, 10
    'ffn_input_dim': args.input_dim,      # input dim of the FFN
    'ffn_hidden_dim': args.hidden_dim,    # hidden dim of the FFN
    'ffn_output_dim': 1801                # output dim of the FFN
        }

# Training parameters (gotten from arguments)

NB_EPOCHS = args.epochs           # 5, 10
BATCH_SIZE = args.batch_size      # 32, 64
FINETUNING = args.finetuning      # False, True

# ffn_num_layers = args_d["ffn_num_layers"]
# ffn_input_dim = args_d["ffn_input_dim"]
# ffn_hidden_dim = args_d["ffn_hidden_dim"]



# -------- #
# Datasets #
# -------- #


# DATASET_FOLDER = "/datasets"
DATASET_FOLDER = "/storage/smiles2spec_data"


if INPUTS == "selfies":
    MODE = "with_selfies_"
elif INPUTS == "smiles":
    MODE = ""


# Use the keep_in_memory=True, since the dataset folder is in read-only

train_dataset = load_from_disk(os.path.join(DATASET_FOLDER, f"train_{MODE}{DATA_TYPE}.hf"), keep_in_memory=True)
val_dataset = load_from_disk(os.path.join(DATASET_FOLDER, f"val_{MODE}{DATA_TYPE}.hf"), keep_in_memory=True)

test_dataset_comp = load_from_disk(os.path.join(DATASET_FOLDER, f"test_{MODE}comp.hf"), keep_in_memory=True)
test_dataset_exp = load_from_disk(os.path.join(DATASET_FOLDER, f"test_{MODE}exp.hf"), keep_in_memory=True)


train_dataset = train_dataset.rename_column("spectrum", "labels")
val_dataset = val_dataset.rename_column("spectrum", "labels")

test_dataset_comp = test_dataset_comp.rename_column("spectrum", "labels")
test_dataset_exp = test_dataset_exp.rename_column("spectrum", "labels")



# --------- #
# Tokenizer #
# --------- #


# Models at https://huggingface.co/DeepChem
#     or at https://huggingface.co/seyonec/ 

HF_TOKEN = "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" # your HF token
login(HF_TOKEN)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=RESULTS_FOLDER) # for ChemBERTa

if MODEL_NAME.startswith("ncfrey/ChemGPT"):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def tokenize(batch, inputs_type="smiles"):
        
    tokens = tokenizer(batch[inputs_type], 
                       truncation=True, 
                       padding=True, 
                       max_length=512)

    return tokens


train_dataset = train_dataset.map(tokenize, fn_kwargs={"inputs_type": INPUTS}, batched=True)
val_dataset = val_dataset.map(tokenize, fn_kwargs={"inputs_type": INPUTS}, batched=True)

test_dataset_comp = test_dataset_comp.map(tokenize, fn_kwargs={"inputs_type": INPUTS}, batched=True)
test_dataset_exp = test_dataset_exp.map(tokenize, fn_kwargs={"inputs_type": INPUTS}, batched=True)


train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

test_dataset_comp.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset_exp.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])



# ----- #
# Model #
# ----- #

MODEL_NAME, MODEL_SUFFIX

num_labels = len(train_dataset[0]["labels"])
num_labels


# * WARNING: replace args by args_d if modified from LLM_Base_Model.ipynb *
class Smile2Spec(nn.Module):
    """A Smile2Spec model contains a LLM head, followed by a Feed Forward MLP."""
    
    def __init__(self, args_d): # args_d and not args!
        """
        Initializes the Smile2Spec model.
        :param args_d: argument for building the model."""

        super(Smile2Spec, self).__init__()

        # # Create LLM head. # xxx old
        # self.LLM = AutoModelForSequenceClassification.from_pretrained(args_d.get('model_name'), 
        #                                                               num_labels=args_d.get('ffn_output_dim'))
        
        if args_d.get('model_name').startswith("ncfrey/ChemGPT"): # xxx
            self.LLM.config.pad_token_id = self.LLM.config.eos_token_id
        
        # Create output objects
        self.output_activation = args_d.get('output_activation')
        self.norm_range = args_d.get('norm_range')

        # Create FFN params
        dropout = nn.Dropout(args_d.get('dropout'))
        activation = args_d.get('activation')

        # Create LLM and FFN layers
        if args_d.get('ffn_num_layers') == 0: # xxx new
            # xxx new
            self.LLM = AutoModelForSequenceClassification.from_pretrained(args_d.get('model_name'), 
                                                                          num_labels=args_d.get('ffn_output_dim'))
            ffn = [activation] # xxx new
        
        if args_d.get('ffn_num_layers') >= 1:
            
            # xxx new
            self.LLM = AutoModelForSequenceClassification.from_pretrained(args_d.get('model_name'), 
                                                                          num_labels=args_d.get('ffn_hidden_dim'))
            
            exactly_one_layer = args_d.get('ffn_num_layers') == 1
            output_dim = args_d.get('ffn_output_dim') if exactly_one_layer else args_d.get('ffn_hidden_dim')
            
            ffn = [
                activation, # xxx new
                # dropout,
                nn.Linear(args_d.get('ffn_hidden_dim'), output_dim), # xxx new architecture
                dropout,    # xxx new
                activation, # xxx new
                ]
            
            # xxx new
            if args_d.get('ffn_num_layers') > 1:
            
                for _ in range(args_d.get('ffn_num_layers') - 2):
                    ffn.extend([
                        # activation,
                        # dropout,
                        nn.Linear(args_d.get('ffn_hidden_dim'), args_d.get('ffn_hidden_dim')),
                        dropout,   # xxx new
                        activation # xxx new
                        ])

                ffn.extend([
                    # activation,
                    # dropout,
                    nn.Linear(args_d.get('ffn_hidden_dim'), args_d.get('ffn_output_dim')),
                    dropout,   # xxx new
                    activation # xxx new
                ])

        self.ffn = nn.Sequential(*ffn)

    def forward(self,
                input_ids = None,
                attention_mask = None,
                labels = None):
        """
        Runs the Smile2Spec model on input.
        
        :return: Output of the Smile2Spec model."""

        # Compute LLM output
        LLM_output = self.LLM(input_ids, attention_mask=attention_mask).logits # type: ignore

        # Compute ffn output
        if args_d.get('ffn_num_layers') > 0 :
            output = self.ffn(LLM_output) 
        else:
            output = LLM_output

        # Positive value mapping
        if self.output_activation == 'exp':
            output = torch.exp(output)
            
        if self.output_activation == 'ReLU':
            f = nn.ReLU()
            output = f(output)

        # Normalization mapping
        if self.norm_range is not None:
            norm_data = output[:, self.norm_range[0]:self.norm_range[1]]
            norm_sum = torch.sum(norm_data, 1)
            norm_sum = torch.unsqueeze(norm_sum, 1)
            output = torch.div(output, norm_sum)

        return output


class SIDLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, model_spectra, target_spectra):

        loss = torch.ones_like(target_spectra)

        loss = torch.mul(torch.log(torch.div(model_spectra, target_spectra)), model_spectra) \
                + torch.mul(torch.log(torch.div(target_spectra, model_spectra)), target_spectra)
        
        loss = torch.sum(loss, dim=1)

        return loss.mean()


# --------- #
# Training #
# --------- #

model = Smile2Spec(args_d)
print(model) # XXX

total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total Params. : ", f"{total_params:,}")
print("Total Trainable Params. : ", f"{total_trainable_params:,}")


class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
   
    def compute_loss(self, model, inputs, return_outputs=False):
        
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss_fct = SIDLoss()
        loss = loss_fct(outputs, labels)
        
        return (loss, {"label": outputs}) if return_outputs else loss


training_args = TrainingArguments(
    
    # output
    output_dir=RESULTS_FOLDER,          
    
    # params
    num_train_epochs=NB_EPOCHS,               # nb of epochs
    per_device_train_batch_size=BATCH_SIZE,   # batch size per device during training
    per_device_eval_batch_size=BATCH_SIZE,    # cf. paper Sun et al.
    learning_rate=5e-5, #2e-5,                # cf. seyonechithrananda / cf. paper Sun et al.
#     warmup_steps=500,                         # number of warmup steps for learning rate scheduler
    warmup_ratio=0.1,                         # cf. paper Sun et al.
    weight_decay=0.01,                        # strength of weight decay
    
    # eval
    eval_strategy="steps",                    # cf. paper Sun et al.
    eval_steps=400,                           # cf. paper Sun et al.
    
    # log
    logging_dir=RESULTS_FOLDER+'logs',  
    logging_strategy='steps',
    logging_steps=400,
    
    # save
    save_strategy='steps',
    save_total_limit=2,
    save_steps=400,                           # save model at every eval (default 500)
    load_best_model_at_end=True,              # cf. paper Sun et al.
    metric_for_best_model='eval_loss',
    # metric_for_best_model='mse', # XXX
    
    report_to="none",                         # "wandb" or "none" to turn wandb off!
    # run_name=f"{model_suffix}",               # name of the W&B run (optional)

    remove_unused_columns=False
)


trainer = CustomTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)


start_time = time.time()
results = trainer.train()
training_time = time.time() - start_time



# save history (loss), trainable params, training time

with open(os.path.join(RESULTS_FOLDER, "log_history.pkl"), "wb") as fh:

    pickle.dump(trainer.state.log_history, fh)
    
    
with open(os.path.join(RESULTS_FOLDER, "training_time.pkl"), "wb") as fh:

    pickle.dump(training_time, fh)
    

with open(os.path.join(RESULTS_FOLDER, "nb_parameters.pkl"), "wb") as fh:
    
    params = {"total_params": total_params, "total_trainable_params" : total_trainable_params}
    
    pickle.dump(params, fh)


torch.save(model.state_dict(), RESULTS_FOLDER + "/model.pt")
print("Model saved.")

# remove checkpoints since best model saved (saves space)

dirs = os.listdir(RESULTS_FOLDER)
dirs = [d for d in dirs if d.startswith("checkpoint")] # checkpoints dirs

for d in dirs:
    shutil.rmtree(os.path.join(RESULTS_FOLDER, d))


# Fine-tune on experimental

if FINETUNING:
    
    #Load dataset
    train_dataset_exp = load_from_disk(os.path.join(DATASET_FOLDER, "train_exp"), keep_in_memory=True)
    val_dataset_exp = load_from_disk(os.path.join(DATASET_FOLDER, "val_exp"), keep_in_memory=True)
    
    #Preprocess Dataset
    train_dataset_exp = train_dataset_exp.rename_column("spectrum", "labels")
    val_dataset_exp = val_dataset_exp.rename_column("spectrum", "labels")
    
    train_dataset_exp = train_dataset_exp.map(tokenize, batched=True)
    val_dataset_exp = val_dataset_exp.map(tokenize, batched=True)
    
    train_dataset_exp.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset_exp.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    
    # training args
    training_args_exp = TrainingArguments(
        
        # output
        output_dir=f"/storage/smiles2spec_models/exp/{MODEL_SUFFIX}",          
        
        # params
        num_train_epochs=NB_EPOCHS // 2,          # nb of epochs
        per_device_train_batch_size=BATCH_SIZE,   # batch size per device during training
        per_device_eval_batch_size=BATCH_SIZE,    # cf. paper Sun et al.
        learning_rate=5e-5, #2e-5,                # cf. seyonechithrananda / cf. paper Sun et al.
    #     warmup_steps=500,                         # number of warmup steps for learning rate scheduler
        warmup_ratio=0.1,                         # cf. paper Sun et al.
        weight_decay=0.01,                        # strength of weight decay

        # eval
        eval_strategy="steps",                    # cf. paper Sun et al.
        eval_steps=400,                           # cf. paper Sun et al.

        # log
        logging_dir=RESULTS_FOLDER+'logs',  
        logging_strategy='steps',
        logging_steps=400,

        # save
        save_strategy='steps',
        save_total_limit=2,
        save_steps=400,                           # save model at every eval (default 500)
        load_best_model_at_end=True,              # cf. paper Sun et al.
        metric_for_best_model='eval_loss',
        # metric_for_best_model='mse', # XXX

        report_to="none",                         # "wandb" or "none" to turn wandb off!
        # run_name=f"{model_suffix}",               # name of the W&B run (optional)

        remove_unused_columns=False
    )

    
    # trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args_exp,
        tokenizer=tokenizer,
        train_dataset=train_dataset_exp,
        eval_dataset=val_dataset_exp
    )
    
    
    # training
    start_time = time.time()
    results = trainer.train()
    training_time = time.time() - start_time


    # save history (loss), trainable params, training time

    with open(os.path.join(RESULTS_FOLDER, "log_history.pkl"), "wb") as fh:

        pickle.dump(trainer.state.log_history, fh)


    with open(os.path.join(RESULTS_FOLDER, "training_time.pkl"), "wb") as fh:

        pickle.dump(training_time, fh)


    with open(os.path.join(RESULTS_FOLDER, "nb_parameters.pkl"), "wb") as fh:

        params = {"total_params": total_params, "total_trainable_params" : total_trainable_params}

        pickle.dump(params, fh)


    torch.save(model.state_dict(), RESULTS_FOLDER + "/model.pt")
    

# ------- #
# Results #
# ------- #


# compute predictions

predicts_comp = trainer.predict(test_dataset_comp)
predicts_exp = trainer.predict(test_dataset_exp)

test_preds_comp, test_truths_comp = predicts_comp.predictions, predicts_comp.label_ids
test_preds_exp, test_truths_exp = predicts_exp.predictions, predicts_exp.label_ids


# save results (if not in loading mode)

torch.save(test_preds_comp, os.path.join(RESULTS_FOLDER,'test_preds_comp.pt'))
torch.save(test_truths_comp, os.path.join(RESULTS_FOLDER,'test_truths_comp.pt'))

torch.save(test_preds_exp, os.path.join(RESULTS_FOLDER,'test_preds_exp.pt'))
torch.save(test_truths_exp, os.path.join(RESULTS_FOLDER,'test_truths_exp.pt'))

print("Predictions saved.")
print("*** Experiment finished ***\n")
