## Get Started:
1. `cd RWKV-LoRA-CPN`
2. `bash scripts/build_env.sh`
3. `bash scripts/convert_dataset.sh example/overfitting`
4. `bash scripts/start_train.sh` (to overfit with large model run `bash scripts/overfit.sh` or for smaller model run `bash scripts/overfit-small.sh`)


## Chat:
1. `bash scripts/chat.sh` (or `bash scripts/chat-small.sh YOUR_MODEL_PATH YOUR_LORA_PATH` if you are using the small model.)

In the 2nd bash: if don't uese conda, please 'pip install -r requirements.txt' directly.
In RWKV-v4neo/src/dataset.py, you need to annotation print(req_len) && print(self.data_size)


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

## NOTICE when u use cloud server
1. when use gpushare.com and can't git/wget or time out, please try `export https_proxy=http://turbo.gpushare.com:30000 http_proxy=http://turbo.gpushare.com:30000` and when u finish download, please `unset http_proxy && unset https_proxy`
2. when use autoDL and can't git/wget or time out, please try `source /etc/network_turbo` and when u finish download, please `unset http_proxy && unset https_proxy`
3. gpushare.com reference: https://gpushare.com/docs/instance/network_turbo/
4. autoDL reference: https://www.autodl.com/docs/network_turbo/

## REFERENCE
1. https://www.codewithgpu.com/i/Blealtan/RWKV-LM-LoRA/RWKV-LM-LoRA
2. https://www.bilibili.com/read/cv22445881
3. https://zhuanlan.zhihu.com/p/629809101