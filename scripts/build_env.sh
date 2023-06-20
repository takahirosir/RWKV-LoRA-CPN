#!/bin/bash
# conda create -n rwkv
# conda activate rwkv
# conda install python=3.10 # the highest python version that pytorch supports as of today
# conda install pip
pip install -r requirements.txt

cd gpt-neox-RWKV
pip install -r requirements.txt && pip install -r requirements-onebitadam.txt && \
    pip install -r requirements-sparseattention.txt && \
    pip install protobuf==3.20.* && \
    pip cache purge
