#!/bin/bash

# Create a virtual environment
python -m venv mlenv

# Activate and install dependencies inside the env
source mlenv/bin/activate
pip install -r requirements.txt


## Create models directory
mkdir _models


## Python Conversion
cd Notebook
jupytext *.ipynb --to py

## Run the scripts 
python notebook_Classification.py
python notebook_Regression.py


## Deactivate after installation
deactivate

echo "Virtual environment 'mlenv' created and dependencies installed."