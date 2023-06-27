#Path change: input, 
cd json2binidx_tool
python tools/preprocess_data.py \
--input /root/code/RWKV-LoRA-CPN/example/overfitting.jsonl \
--output-prefix /root/code/RWKV-LoRA-CPN/datasets/ \
--vocab ./20B_tokenizer.json \
--dataset-impl mmap \
--tokenizer-type HFTokenizer \
--append-eod
