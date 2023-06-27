## Get Started:
1. `cd RWKV-LoRA-CPN`
2. `bash scripts/build_env.sh`
3. `bash scripts/download_model_and_dataset.sh`
4. `bash scripts/convert_dataset.sh example/overfitting`
5. `bash scripts/start_train.sh`

now the 4th bash has something wrong with it
and in the 2nd bash: if don't uese conda, please 'pip install -r requirements.txt' directly.

## REFERENCE
1. https://www.codewithgpu.com/i/Blealtan/RWKV-LM-LoRA/RWKV-LM-LoRA
2. https://www.bilibili.com/read/cv22445881
3. https://zhuanlan.zhihu.com/p/629809101


## NOTICE in scripts/start_train.sh: 
1. ctx_len represent the tokens number in jsonl, we change it into 70 for satisfy the number of self.data_size is bigger than req_len(in dataset.py)
2. epoch_steps 1000 for every epoch
3. epoch_count 10 is useless
4. epoch_begin * when load a lora model you can begin in next number
5. epoch_save 1 every epoch will be saved as pth file 'rwkv-*.pth'
6. lora_alpha 32 represent the learning rate?
7. warmup_steps 50 when you load a lora model, this number can be 50 
8. lora_dropout 0.0 for overfitting, the number is smaller, then it will be more overfitting
9. lora_load /root/autodl-tmp/lora_checkpoints/rwkv-17.pth the loading model's path

## NOTICE in chat.py
1. CHAT_LANG = 'Chinese' you can chose English or Chinese
2. args.MODEL_NAME = '*'  this should be the loading RWKV model path
3. args.MODEL_LORA = '*'  this should be the loading lora model path
4. args.lora_r = 8
5. args.lora_alpha = 32
6. BEFORE RUN chat.py, PLEASE RUN export RWKV_JIT_ON=1 
