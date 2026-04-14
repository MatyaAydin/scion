#!/bin/bash

pip install -r requirements.txt
pip install -r data/requirements.txt
pip install optuna
pip install optuna-integration[pytorch_distributed]
pip install --upgrade optuna

torchrun --standalone --nproc_per_node=4 train_gpt_optuna.py