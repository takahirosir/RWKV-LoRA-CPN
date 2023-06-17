#!/bin/bash
source scripts/common.sh

# cd  ~/train/RWKV-LM-LoRA/RWKV-v4neo
cd RWKV-v4neo
python train.py \
    --load_model "$MODEL_PATH/RWKV-4-Raven-3B-v10x-Eng49-Chn50-Other1-20230423-ctx4096.pth" \
    --proj_dir "/proj_dir/OutBinIdx" \
    --data_file "$DATASET_PATH/databricks-dolly-15k.jsonl" \
    --data_type utf-8 \
    --vocab_size 50277 \
    --ctx_len 4096 \
    --epoch_steps 1000 \
    --epoch_count 1000 \
    --epoch_begin 0 \
    --epoch_save 5 \
    --micro_bsz 2 \
    --n_layer 24 \
    --n_embd 1024 \
    --pre_ffn 0 \
    --head_qk 0 \
    --lr_init 1e-4 \
    --lr_final 1e-4 \
    --warmup_steps 0 \
    --beta1 0.9 \
    --beta2 0.999 \
    --adam_eps 1e-8 \
    --accelerator gpu \
    --devices 1 \
    --precision bf16 \
    --strategy deepspeed_stage_2 \
    --grad_cp 0 \
    --lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.01 \
    --lora_parts=att
