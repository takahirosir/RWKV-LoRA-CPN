Explanation

train.py

1. --load_model	RWKV预训练的底模型.pth （需要带上.pth后缀）
2. --proj_dir /path/save_pth	最后训练好的pth保存的路径，注意不要以 / 结尾，否则会出现找不到目录
3. --data_file /path/file	预处理语料数据（如果是binidx语料，不需要加bin或idx，只要文件名）
4. --data_type binidx	语料的格式，目前支持"utf-8", "utf-16le", "numpy", "binidx", "dummy", "wds_img", "uint16"
5. --vocab_size 50277	词表大小，格式转换的时候，最后返回的一个数值，txt数据的时候可以设置为0表示自动计算
6. --ctx_len 1024（看显存，越大越好，模型文件名有最大ctx_len） 
7. --accumulate_grad_batches 8  （貌似已废用）
8. --epoch_steps 1000  指每个epoch跑1000步	
9. --epoch_count 20 指跑20个epoch，但是在rwkv中不会自动停止训练，需要人工停止，所以这个参数没什么大作用
10. --epoch_begin 0	epoch开始值，表示从第N个epoch开始加载
11. --epoch_save 5	训练第几轮开始自动保存，5表示每5轮epoch自动保存一个pth模型文件
12. --micro_bsz 1	  微型批次大小（每个GPU的批次大小）（改大应该会更好些，显存占用更大） 
13. --n_layer 32（看模型，Pile上有介绍） 
14. --n_embd 2560（看模型，Pile上有介绍） 
15. --pre_ffn 0   用ffn替换第一个att层（有时更好）
16. --head_qk 0 	headQK技巧
17. --lr_init 6e-4	6e-4表示L12-D768，4e-4表示L24-D1024，3e-4表示L24-D2048
18. --lr_final 1e-5 
19. --warmup_steps 0 预热步骤，如果你加载了一个模型，就试试50
20. --beta1 0.9 
21. --beta2 0.999  当你的模型接近收敛时使用0.999
22. --adam_eps 1e-8 
23. --accelerator gpu  目前有gpu、cpu，但是正常情况下cpu是无法支持training的
24. --devices 1 （单卡为1，多卡就按照对应的卡数来填写，比如8卡就填写8） 
25. --precision fp16  策略精度，目前支持"fp32", "tf32", "fp16", "bf16"
26. --strategy deepspeed_stage_2  这是lightning吃的策略参数，顾名思义是deepspeed的stage 2
27. --grad_cp 1 （开启加速） 配置这个显存量，0应该可以直接全量 
28. --lora   开启lora训练
29. --lora_r 8 		r 越多，可训练参数越多
30. --lora_alpha 32 	alpha 越大，可以看作等效学习率越大
31. --lora_dropout 0.01	dropout用来防过拟合
32. --lora_parts=att,ffn,time,ln   这里att, ffn, time 和 ln指的是TimeMix, ChannelMix, time decay/first/mix参数, layernorm参数；
33. --lora_load /path/lora.pth  是指你已lora训练好的pth文件，如果想继续之前已lora训练好的pth上继续Lora训练，那么这里就填写pth对应的路径即可。如果没有则删除这个参数即可。