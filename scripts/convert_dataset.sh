#!/bin/bash
source scripts/common.sh

# Vars
INPUT_DIR=${1}
JSONL_FILEPATH=$DATASET_DIR/dataset.jsonl
JSON2BINIDX_DIR=$PWD/json2binidx_tool

# Create dirs
mkdir -p $DATASET_DIR

# Convert txt files to jsonl
rm $JSONL_FILEPATH
python tools/txt2jsonl.py \
--input $INPUT_DIR \
--output $JSONL_FILEPATH

# Convert jsonl file to binidx
echo Converting jsonl file $JSONL_FILEPATH into binidx...
cd $JSON2BINIDX_DIR

python tools/preprocess_data.py \
--input $JSONL_FILEPATH \
--output-prefix $DATASET_DIR/ \
--vocab ./20B_tokenizer.json \
--dataset-impl mmap \
--tokenizer-type HFTokenizer \
--append-eod
