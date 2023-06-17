#!/bin/bash
conda create -n rwkv
conda activate rwkv
conda install python=3.10 # the highest python version that pytorch supports as of today
conda install pip
pip install -r requirements.txt
