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

mv *.py ../Script
cd ../Script
mv notebook_Classification.py train_Classification.py
mv notebook_Regression.py train_Regression.py

cp train_Classification.py predict_Classification.py
cp train_Regression.py predict_Regression.py

## Run the scripts 
python train_Classification.py
python train_Regression.py


## Deactivate after installation
deactivate

echo "Virtual environment 'mlenv' created and dependencies installed."