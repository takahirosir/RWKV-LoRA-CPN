#!/bin/bash
RWKV_DIR=RWKV-v4neo
NEOX_DIR=gpt-neox-RWKV
INPUT_DIR=$PWD/example
JSONL_FILEPATH=$PWD/datasets/noval_example.jsonl
BINIDX_DIR=$PWD/datasets/binidx

echo Converting txt files into jsonl files...
python $NEOX_DIR/txt2jsonl.py \
--input=$INPUT_DIR \
--output=$JSONL_FILEPATH
echo Conversion done, output path: $JSONL_FILEPATH

echo Converting jsonl file $JSONL_FILEPATH into binidx...
python3 $NEOX_DIR/tools/preprocess_data.py \
--input $JSONL_FILEPATH \
--output-prefix $BINIDX_DIR/ \
--vocab $RWKV_DIR/20B_tokenizer.json \
--dataset-impl mmap \
--tokenizer-type HFTokenizer \
--append-eod
echo Conversion done, output path: $BINIDX_DIR
