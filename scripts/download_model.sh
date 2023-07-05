#!/bin/bash
source scripts/common.sh

# The local model path to be updated and used after downloading.
LOCAL_MODEL_PATH=""

# A function to download a model (.pth) file from a given url.
# It will only download if the local model is not available.
function download_model_to_local() {
    local model_url=${1}
    # Assign the first parameter passed in to the local variable "model_url", which is the download link of the model.

    # Extract the file name from the given url.
    local filename=${model_url##*/}
    # Compose the path to the local model file.
    # The var is global.
    LOCAL_MODEL_PATH=$MODEL_DIR/$filename
    # Check if the local model file exits.
    if [ ! -f $LOCAL_MODEL_PATH ]; then
    # if there is no model file in the local path, then run the following code
        echo local model file $LOCAL_MODEL_PATH does not exits, downloading from $MODEL_URL...
        # Print in the shell to prompt that the model is not exits

        # Adding the following line to avoid the SSL issue during wget.
        # Source: https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models
        CURL_CA_BUNDLE=""
        # Create model folder and download model.
        mkdir -p $MODEL_DIR
        # Download the model to the local folder.
        wget $model_url -P $MODEL_DIR/
        if [ ! -f $LOCAL_MODEL_PATH ]; then
        # if finish download but there is still no model in the dir
            print_warning "Download ${model_url} failed!"
            exit
        fi
        echo model $MODEL_URL saved to $LOCAL_MODEL_PATH
        # print the prompt information that the model is saved to dir
    fi
}


# Create dataset folder and download dataset
mkdir -p $DATASET_DIR
# wget https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl -P $DATASET_PATH/
# This data(jsonl) file is useless

# Actually in AutoDL/gupshare ect. domestic GPU web, there will be like such "Unable to establish SSL connection." problem
# Please refer to the following method
# AutoDL https://www.autodl.com/docs/network_turbo/
# GPUshare https://gpushare.com/docs/instance/network_turbo/