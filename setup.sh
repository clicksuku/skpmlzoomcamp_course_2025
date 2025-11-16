#!/bin/bash

# Create a virtual environment
python -m venv mlenv

# Activate and install dependencies inside the env
source mlenv/bin/activate
pip install -r requirements.txt


## Create models directory
mkdir _models

## Python Conversion
jupytext SKP_MidTerm_Project_Regression.ipynb --to py
jupytext SKP_MidTerm_Project_Classification.ipynb --to py

## Run the scripts 
python SKP_MidTerm_Project_Regression.py
python SKP_MidTerm_Project_Classification.py


## Deactivate after installation
deactivate

echo "Virtual environment 'mlenv' created and dependencies installed."