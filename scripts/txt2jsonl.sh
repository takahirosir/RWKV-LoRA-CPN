#!/bin/bash
RWKV_DIR=$PWD/RWKV-v4neo
JSON2BINIDX_DIR=json2binidx_tool
INPUT_DIR=$PWD/example
DATASET_DIR=$PWD/datasets
JSONL_FILEPATH=$DATASET_DIR/noval_example.jsonl
BINIDX_DIR=$DATASET_DIR/binidx

# Create dirs
mkdir -p $DATASET_DIR

# Convert txt files to jsonl
python tools/txt2jsonl.py \
--input=$INPUT_DIR \
--output=$JSONL_FILEPATH

# Convert jsonl file to binidx
echo Converting jsonl file $JSONL_FILEPATH into binidx...
cd $JSON2BINIDX_DIR

python tools/preprocess_data.py \
--input $JSONL_FILEPATH \
--output-prefix $BINIDX_DIR \
--vocab ./20B_tokenizer.json \
--dataset-impl mmap \
--tokenizer-type HFTokenizer \
--append-eod
