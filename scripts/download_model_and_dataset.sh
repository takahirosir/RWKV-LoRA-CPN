#!/bin/bash
source scripts/common.sh

# Adding the following line to avoid the SSL issue during wget
# Source: https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models
CURL_CA_BUNDLE=""
# Create model folder and download model
mkdir -p $MODEL_DIR
MODEL_URL=https://huggingface.co/BlinkDL/rwkv-4-pile-3b/resolve/main/RWKV-4-Pile-3B-Chn-testNovel-done-ctx2048-20230312.pth
# MODEL_URL=https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/RWKV-4-Raven-3B-v10x-Eng49%25-Chn50%25-Other1%25-20230423-ctx4096.pth
wget $MODEL_URL -P $MODEL_DIR/


# Create dataset folder and download dataset
mkdir -p $DATASET_DIR
# wget https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl -P $DATASET_PATH/
# This data(jsonl) file is useless