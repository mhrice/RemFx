#! /bin/bash

# Example usage:
# scripts/eval.sh remfx_detect 0-0
# scripts/eval.sh distortion_aug 0-0 -ckpt logs/ckpts/2023-01-21-12-21-44
# First 2 arguments are required, third argument is optional

# Default value for the optional parameter
ckpt_path=""
export DATASET_ROOT=RemFX_eval_datasets
# Function to display script usage
function display_usage {
    echo "Usage: $0 <experiment> <dataset> [-ckpt {ckpt_path}]"
}

# Check if the number of arguments is less than 2 (minimum required)
if [ "$#" -lt 2 ]; then
    display_usage
    exit 1
fi

dataset_name=$2

# Parse optional parameter if provided
if [ "$3" == "-ckpt" ]; then
    # Check if the ckpt_path is provided
    if [ -z "$4" ]; then
        echo "Error: -ckpt flag requires a path argument."
        display_usage
        exit 1
    fi
    ckpt_path="$4"
fi

# If ckpt_path is empty, run chain inference
if [ -z "$ckpt_path" ]; then
    echo "Running chain inference"
    python scripts/chain_inference.py +exp=$1 datamodule.train_dataset=None datamodule.val_dataset=None datamodule.test_dataset.render_root=./RemFX_eval_datasets/ render_files=False num_removed_effects=[${dataset_name:0:1},${dataset_name:2:1}]
    exit 1
fi


# Otherwise run inference on the specified checkpoint
echo "Running monolithic inference on checkpoint $3"
python scripts/test.py +exp=$1 datamodule.train_dataset=None datamodule.val_dataset=None datamodule.test_dataset.render_root=./RemFX_eval_datasets/ datamodule.test_dataset.num_kept_effects="[0,0]" num_removed_effects=[${dataset_name:0:1},${dataset_name:2:1}] effects_to_keep=[] effects_to_remove="[distortion, compressor,reverb,chorus,delay]" render_files=False +ckpt_path=$ckpt_path



