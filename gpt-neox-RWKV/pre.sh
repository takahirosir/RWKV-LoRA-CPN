python3 tools/preprocess_data.py \
--input /root/autodl-tmp/jsonl/text.jsonl \
--output-prefix /root/autodl-tmp/data/ \
--vocab ./20B_tokenizer.json \
--dataset-impl mmap \
--tokenizer-type HFTokenizer \
--append-eod