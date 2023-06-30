#!/bin/bash
source scripts/download_model.sh

# Model card: https://huggingface.co/BlinkDL/rwkv-4-pile-169m
download_model_to_local https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4b-Pile-171M-20230202-7922.pth

# Start training
cd RWKV-v4neo
# Introduction of the parameters: https://zhuanlan.zhihu.com/p/629809101
python3 ./train.py \
--load_model $LOCAL_MODEL_PATH \
--proj_dir $PROJECT_DIR/lora_checkpoints \
--data_file $DATASET_DIR/_text_document \
--data_type binidx \
--vocab_size 50277 \
--ctx_len 70 \
--accumulate_grad_batches 8 \
--epoch_steps 1000 \
--epoch_count 10 \
--epoch_begin 18 \
--epoch_save 1 \
--micro_bsz 1 \
--n_layer 12 \
--n_embd 768 \
--pre_ffn 0 \
--head_qk 0 \
--lr_init 1e-3 \
--lr_final 1e-3 \
--warmup_steps 0 \
--beta1 0.9 \
--beta2 0.999 \
--adam_eps 1e-8 \
--accelerator gpu \
--devices 1 \
--precision fp32 \
--strategy ddp_find_unused_parameters_false \
--grad_cp 1 \
--lora \
--lora_r 8 \
--lora_alpha 32 \
--lora_dropout 0.0 \
--lora_parts=att,ffn,time,ln \
# --lora_load /root/autodl-tmp/lora_checkpoints/rwkv-17.pth
# some args form 'acceleretor' to 'grad_cp' is in the module pytorch_lighting and they are USEFULL! 
