#!/bin/bash
# PWD(print working directory)
RWKV_DIR=$PWD/RWKV-v4neo
MODEL_DIR=$PWD/models
DATASET_DIR=$PWD/datasets
PROJECT_DIR=$PWD/project

# Printers
function print_warning() {
    echo -e "\e[1;31m${1} \e[0m"
}
