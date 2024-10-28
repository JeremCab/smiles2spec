# ======= #
# Results #
# ======= #


# ---------- #
# Librairies #
# ---------- #


# Libraries to be installed before runnin the script
# pip install rdkit
# pip install transformers==4.45.2 # 4.46.0 makes pb
# pip install -U huggingface_hub


import os
import pickle
import json

import argparse

import pandas as pd
import numpy as np
import torch

from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error

# from datasets import Dataset
from datasets import load_from_disk

from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers.data.data_collator import DataCollatorWithPadding

from transformers.modeling_outputs import SequenceClassifierOutput

from rdkit import Chem # type: ignore

import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
# plt.style.use('ggplot')

from huggingface_hub import login

import torch.nn as nn


# -------- #
# Preamble #
# -------- #


# model_names_original = [
                        # "DeepChem/ChemBERTa-5M-MLM", 
                        # "DeepChem/ChemBERTa-5M-MTR",                # for SMILES
                        # "DeepChem/ChemBERTa-10M-MLM",
                        # "DeepChem/ChemBERTa-10M-MTR",
                        # "DeepChem/ChemBERTa-77M-MLM",
                        # "DeepChem/ChemBERTa-77M-MTR",
                        # "seyonec/BPE_SELFIES_PubChem_shard00_160k"  # for SELFIES
                        # "ncfrey/ChemGPT-4.7M"                       # for SELFIES
                        # ]


HF_TOKEN = "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" # your HF token
login(HF_TOKEN)


# 0. Create argument parser
parser = argparse.ArgumentParser(description="Training arguments...")
parser.add_argument("--results_folder", type=str, help="Results folder.") # non-default
parser.add_argument("--results_dir", type=str, default="/storage/smiles2spec_models", help="Results directory.")
parser.add_argument("--data_dir", type=str, default="/storage/smiles2spec_data", help="Data directory.")
parser.add_argument("--new_results_dir", type=str, default="/notebooks/smiles2spec/notebooks/results", help="New results folder.")
args = parser.parse_args()

# 1. Select results dir
results_folder = args.results_folder
RESULTS_DIR = args.results_dir
NEW_RESULTS_DIR = args.new_results_dir
RESULTS_FOLDER = os.path.join(RESULTS_DIR, results_folder)         # where to load model's predictions, etc.
RESULTS_FOLDER_BIS = os.path.join(NEW_RESULTS_DIR, results_folder) # where to write new results


if not os.path.exists(RESULTS_FOLDER_BIS):
    os.mkdir(RESULTS_FOLDER_BIS)


# 2. Set parameters
DATA_DIR = args.data_dir

INPUTS = RESULTS_FOLDER.split("_")[1].split("/")[1] # "smiles", "selfies"

if INPUTS == "selfies":
    MODE = "with_selfies_"
elif INPUTS == "smiles":
    MODE = ""
    
DATA_TYPE = RESULTS_FOLDER.split("_")[2] # ["comp", "exp"]

if DATA_TYPE == "comp":
    test_size = 8551
    
elif DATA_TYPE == "exp":
    test_size = 6000
    
elif DATA_TYPE == "comp2exp":
    test_size = 6000
    
MODEL_NAME = RESULTS_FOLDER.split("_")[3] # ["ChemBERTa-5M-MTR", "ChemGPT-4.7M"]

print(f"Inputs:\t\t {INPUTS}")
print(f"Data type:\t {DATA_TYPE}")
print(f"Model:\t\t {MODEL_NAME}")
print(f"Results foldler: {RESULTS_FOLDER}")


# ------------- #
# Plot function #
# ------------- #


def plot_example(test_truths, test_preds, i, data_type, add_info="", conv=True, filename=None):
    """Plot i-th item of (test_truths, test_preds)"""
    
    # get 2D-Mol graph
    test_dataset = load_from_disk(os.path.join(DATA_DIR, f"test_{MODE}{data_type}.hf"))
    smiles = test_dataset[int(i)]["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.Draw.MolToImage(mol)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
        
    # true spectrum
    spectrum_truth = test_truths[i, :]
    #spectrum_truth = gaussian_filter1d(spectrum_truth, sigma=5) # Gaussian convolution
    #spectrum_truth_sum = spectrum_truth.sum()                   # Normalization
    #spectrum_truth = spectrum_truth / spectrum_truth_sum
    
    # predicted spectrum
    spectrum_pred = test_preds[i, :]
    if conv:
        spectrum_pred = gaussian_filter1d(spectrum_pred, sigma=5)  # Gaussian convolution
    #spectrum_pred_sum = spectrum_pred.sum()                       # Normalization
    #spectrum_pred = spectrum_pred / spectrum_pred_sum
    
    xticks = 400 + 2*np.array(range(len(spectrum_truth)))
    plt.plot(xticks, spectrum_truth, label="true spectrum", alpha=1.0, linewidth=2)
    plt.plot(xticks, spectrum_pred, label=f"predicted spectrum{add_info}", alpha=0.6, linewidth=2)
    plt.title(smiles, size=12) # type: ignore
    
    # 2D-Mol graph
    x, y = 2500, spectrum_truth.max() / 1.5
    im = OffsetImage(mol, zoom=0.4, zorder=10)
    ab = AnnotationBbox(im, (x, y), frameon=False)
    ax.add_artist(ab)
    
    plt.xlabel("wavenumbers (cm$^{-1}$)")
    plt.xticks()
    plt.ylabel("absorbance (a.u.)")
    plt.legend()
    plt.title(f"Model trained on {DATA_TYPE}. spectra, predictions on {data_type}. spectra")

    if filename is not None:
        plt.savefig(f"{RESULTS_FOLDER_BIS}/{filename}")
    
    plt.show()


# ~~~~~~~ #
# Example #
# ~~~~~~~ #
# # data_type for predictions (the model is trained of DATA_TYPE (capital))

# data_type = "comp"
# test_preds = torch.load(os.path.join(RESULTS_FOLDER, f'test_preds_{data_type}.pt'))
# test_truths = torch.load(os.path.join(RESULTS_FOLDER, f'test_truths_{data_type}.pt'))
# for i in np.random.choice(test_size, 5):
#     plot_example(test_truths, test_preds, i, data_type=DATA_TYPE)


# ---- #
# Loss # 
# ---- #


# 1. Load loss
file = os.path.join(RESULTS_FOLDER, "log_history.pkl")

history_d = {}

with open(file, "rb") as fh:

    history_d[MODEL_NAME] = pickle.load(fh)

losses_d = {}

l_full = history_d[MODEL_NAME]
l_train = [d for d in l_full if "loss" in d.keys()]
l_val = [d for d in l_full if "eval_loss" in d.keys()]

train_loss = {d["step"] : d["loss"] for d in l_train}
val_loss = {d["step"] : d["eval_loss"] for d in l_val}

losses_d[MODEL_NAME] = (train_loss, val_loss)


# 2. Plot loss
train_loss, val_loss = losses_d[MODEL_NAME]

plt.figure(figsize=(7, 3))
plt.plot(list(train_loss.keys()), list(train_loss.values()), label="train loss", linewidth=2)
plt.plot(list(val_loss.keys()), list(val_loss.values()), label="validation loss", linewidth=2)
plt.xlabel("steps")
plt.ylabel("SID")
plt.title(f"{MODEL_NAME} ({DATA_TYPE}. spectra)")
plt.legend()
plt.savefig(f"{RESULTS_FOLDER_BIS}/loss.pdf", bbox_inches='tight')


# ------- #
# Metrics #
# ------- #


# 1. Load predictions
metrics_d = {}
    
for data_type in ["comp", "exp"]:
    
    metrics_d[data_type] = {}
    
    metrics_d[data_type][MODEL_NAME] = {}

    test_preds = torch.from_numpy(torch.load(os.path.join(RESULTS_FOLDER, f'test_preds_{data_type}.pt')))
    test_truths = torch.from_numpy(torch.load(os.path.join(RESULTS_FOLDER, f'test_truths_{data_type}.pt')))

    metrics_d[data_type][MODEL_NAME]["test_preds"] = np.array(test_preds)
    metrics_d[data_type][MODEL_NAME]["test_truths"] = np.array(test_truths)

    
# 2. Mertrics functions
def get_MSE(y, y_hat):
    """mean squared error (MSE)"""

    MSE = nn.MSELoss(reduction='none')    

    MSEs = torch.mean(MSE(y, y_hat), 1)

    std, mean = torch.std_mean(MSEs)
    
    return MSEs.numpy(), mean.item(), std.item()


def get_RMSE(y, y_hat):
    """root mean squared error (RMSE)"""

    MSE = nn.MSELoss(reduction='none')

    MSEs = torch.mean(MSE(y, y_hat), 1)
    
    RMSEs = torch.sqrt(MSEs)

    std, mean = torch.std_mean(RMSEs)
    
    return RMSEs.numpy(), mean.item(), std.item()


def get_TMSE(y, y_hat):
    """target-weighted mean squared error (TMSE)"""
    TMSEs = torch.mean(((y - y_hat)**2)/y, 1)

    std, mean = torch.std_mean(TMSEs)
    
    return TMSEs.numpy(), mean.item(), std.item()


class SIDLoss(nn.Module):
    
    def __init__(self):
        
        super().__init__()

    def forward(self, model_spectra, target_spectra):

        loss = torch.ones_like(target_spectra)

        loss = torch.mul(torch.log(torch.div(model_spectra, target_spectra)), model_spectra) \
                + torch.mul(torch.log(torch.div(target_spectra, model_spectra)), target_spectra)
        
        loss = torch.sum(loss, dim=1)

        return loss


def get_SIS(y, y_hat, conv=True, norm=True):
    """spectral information similarity (SIS)"""
    
    # Gaussian Convolution
    y_hat_conv = gaussian_filter1d(y_hat, 5)                     # Ok
    y_conv = y.numpy() if not conv else gaussian_filter1d(y, 5)  #

    # Normalization
    if norm:
        y_hat_conv = nn.functional.normalize(torch.tensor(y_hat_conv), p=1)  #
        y_conv = nn.functional.normalize(torch.tensor(y_conv), p=1)          #
    else: # back to torch (not optimal)
        y_hat_conv = torch.from_numpy(y_hat_conv)
        y_conv = torch.from_numpy(y_conv)
    
    # Initialize SID Class
    SID = SIDLoss()
    
    SIDs = SID(y_hat_conv, y_conv)

    SISs = 1/(1+SIDs)

    std, mean = torch.std_mean(SISs)
    
    return SISs.numpy(), mean.item(), std.item()


def get_SID(y, y_hat):
    """spectral information divergence"""

    SID = SIDLoss()
    
    SIDs = SID(y_hat, y)

    std, mean = torch.std_mean(SIDs)

    return SIDs.numpy(), mean.item(), std.item()


# 3. Compute metrics
metrics_df_d = {}

for data_type in ["comp", "exp"]:
    
    # compute metrics
    y = torch.from_numpy(metrics_d[data_type][MODEL_NAME]["test_truths"])
    y_hat = torch.from_numpy(metrics_d[data_type][MODEL_NAME]["test_preds"])

    MSEs, MSE_mean, MSE_std = get_MSE(y, y_hat)
    metrics_d[data_type][MODEL_NAME]["MSEs"] = MSEs
    metrics_d[data_type][MODEL_NAME]["MSE"] = MSE_mean, MSE_std

    RMSEs, RMSE_mean, RMSE_std = get_RMSE(y, y_hat)
    metrics_d[data_type][MODEL_NAME]["RMSEs"] = RMSEs
    metrics_d[data_type][MODEL_NAME]["RMSE"] = RMSE_mean, RMSE_std

    TMSEs, TMSE_mean, TMSE_std = get_TMSE(y, y_hat)
    metrics_d[data_type][MODEL_NAME]["TMSEs"] = TMSEs
    metrics_d[data_type][MODEL_NAME]["TMSE"] = TMSE_mean, TMSE_std

    SIDs, SID_mean, SID_std = get_SID(y, y_hat)
    metrics_d[data_type][MODEL_NAME]["SIDs"] = SIDs
    metrics_d[data_type][MODEL_NAME]["SID"] = SID_mean, SID_std

    SISs, SIS_mean, SIS_std = get_SIS(y, y_hat)      # xxx results significantly degraded if conv/norm is False!
    metrics_d[data_type][MODEL_NAME]["SISs"] = SISs
    metrics_d[data_type][MODEL_NAME]["SIS"] = SIS_mean, SIS_std
    
    
    # Save metrics
    metrics_df = pd.DataFrame.from_dict(metrics_d[data_type], orient="index")
    metrics_df.to_csv(f"{RESULTS_FOLDER_BIS}/metrics_test_{data_type}.csv")
    metrics_df_d[data_type] = metrics_df
    print(f"Metrics computed and saved for {MODEL_NAME} on {data_type} spectra.")

    # metrics_exp_df = pd.DataFrame.from_dict(metrics_d["exp"], orient="index")
    # metrics_exp_df.to_csv(f"{RESULTS_FOLDER_BIS}/metrics_test_exp.csv")
    # print(f"Metrics computed and saved for {MODEL_NAME} on {data_type} spectra.")


# 4. Check metrics (only in notebook)
# metrics_comp_df = metrics_df_d["comp"]
# metrics_comp_df
# metrics_exp_df = metrics_df_d["exp"]
# metrics_exp_df


# ------------- #
# Distributions #
# ------------- #


# 1. Plot fuunction
def plot_distribution(metrics, mean_std, metrics_name=None, model_name=None, data_type=None):
    """Plot distribution of a given metrics and save it as pdf"""
    
    mean, std = mean_std
    
    plt.figure(figsize=(4.5,3))
    plt.hist(metrics, bins=50, density=True, 
             edgecolor='black', 
             linewidth=0.5,
             color="C1",
             alpha=0.5,
             label=f"mean: {mean:.4f}\nstd: {std:.4f}")

    plt.xlabel(f"{metrics_name}")
    plt.legend()
    plt.title(f"{model_name}\n train: {DATA_TYPE}. data, results: {data_type}. test set")
    plt.savefig(f"{RESULTS_FOLDER_BIS}/{metrics_name}_test_{data_type}.pdf", bbox_inches='tight')


# 2. Compute distributions
for data_type in ["comp", "exp"]:

    for model_name, values in metrics_d[data_type].items():

        # for metrics_name in ["MSE", "RMSE", "TMSE", "SID", "SIS"]:
        for metrics_name in ["SIS"]:

            metrics = metrics_d[data_type][model_name][metrics_name+"s"]
            mean_std = metrics_d[data_type][model_name][metrics_name]

            plot_distribution(metrics=metrics, 
                              mean_std=mean_std, 
                              metrics_name=metrics_name, 
                              model_name=model_name,
                              data_type=data_type)


# ------------------------- #
# Percentile-based examples #
# ------------------------- #

GAUSSIAN_CONV = True

for data_type in ["comp", "exp"]:
    
    for model_name, values in metrics_d[data_type].items():

        print(model_name, data_type)

        test_truths = metrics_d[data_type][model_name]["test_truths"]
        test_preds = metrics_d[data_type][model_name]["test_preds"]
        if GAUSSIAN_CONV:
            test_preds = gaussian_filter1d(test_preds, 5) # plot smoothed predictions if preferred

        SISs = metrics_d[data_type][model_name]["SISs"]
        percentiles = []

        for p in range(10, 100, 10):
            percentiles.append(np.percentile(SISs, p))

        lb = 0

        for i in range(len(percentiles)): 

            # get percentile
            ub = percentiles[i]
            mask = (SISs >= lb) & (SISs <= ub)
            lb = percentiles[i]

            # plot example from percentile 
            indices = np.where(mask==True)[0]
            index = np.random.choice(indices, size=1)[0]

            p = i*10 + 10
            print(f"example in percentile {p}")
            filename = f"{model_name}_{data_type}_percentile_{p}.png"
            add_info = f"\nSIS: {SISs[index]:.4f}"
            
            filename = f"example_p_{p}.pdf"
            
            plot_example(test_truths, test_preds, index, data_type, add_info, filename=filename)

