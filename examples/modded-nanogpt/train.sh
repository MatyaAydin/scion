#!/bin/bash

pip install -r requirements.txt
pip install -r data/requirements.txt
torchrun --standalone --nproc_per_node=4 train_gpt_scion.py