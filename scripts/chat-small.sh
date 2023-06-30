#!/bin/bash
source scripts/common.sh
source scripts/overfit.sh

cd RWKV-v4neo
python chat.py \
--load_model $LOCAL_MODEL_PATH \
--load_lora ${1} \
--n_layer 12 \
--n_embd 786 \
--ctx_len 70 \
--lora_r 8 \
--lora_alpha 32
