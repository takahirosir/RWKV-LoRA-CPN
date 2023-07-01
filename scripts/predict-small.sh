#!/bin/bash
source scripts/common.sh

MODEL_PATH=$PWD/${1}
LORA_PATH=""none
LORA_R=0
if [ $# -eq 2 ]; then 
    LORA_PATH=$PWD/${2}
    LORA_R=8
fi

cd RWKV-v4neo
python predict.py \
--load_model ${MODEL_PATH} \
--load_lora ${LORA_PATH} \
--n_layer 12 \
--n_embd 768 \
--ctx_len 70 \
--lora_r $LORA_R \
--lora_alpha 32 \
--precision fp32
