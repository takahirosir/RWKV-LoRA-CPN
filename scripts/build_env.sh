#!/bin/bash

# Fetch sub modules
# Fill the json2binidx_tool cause it's another repo form other's
git submodule update --init

# Build env for RWKV
pip install -r requirements.txt

# Build env for json2binidx tool
cd json2binidx_tool
pip install -r requirements.txt
