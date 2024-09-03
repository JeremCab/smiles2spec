# SMILE2Spec

Predicting IR vibrational spectra from SMILES representation of molecules using LLMs.

## Description

This repository consists of my internship's project in 2024, during my M1 Maths&IA at Université Paris-Saclay. It consists of several notebooks and packages containing various versions of prediction model for vibrationnal spectra.

These models were developed in order to predict IR vibrational spectra of various molecules, encoded in the form of SMILES. It utilizes the recent ChemBERTa models in order to encode these SMILES, then predict the IR vibrational spectrum in the form of an aborbance vector.

## Getting Started

### Dependencies

* Python 3.11 recommended
* Pytorch, HuggingFace, transformers

### Installing

* No modifications are needed, notebooks can be ran directly
* Models and Losses can be directly imported from the smile2spec folder


### Executing program

* Models can be trained directly using the corresponding notebook
* Otherwise, import the base model and SID loss from the corresponding folder in the smile2spec folder, as described in the Example Notebook


## Authors


* [CAUDARD Joris](https://github.com/JorisCaudard)


## Acknowledgments

* [Deepchem](https://github.com/deepchem) : ChemBERTa Model
* [ChemProp](https://github.com/gfm-collab/chemprop-IR) : GNN Model for Vibrational spectra prediction
