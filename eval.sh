#! /bin/bash

# Example usage:
# ./eval.sh remfx_detect 0-0
# ./eval.sh distortion_aug 0-0 -ckpt logs/ckpts/2023-01-21-12-21-44
# First 2 arguments are required, third argument is optional

# Check if first argument is empty
if [ -z "$1" ]
then
  echo "No experiment name supplied"
  exit 1
fi

# Check if second argument is empty
if [ -z "$2" ]
then
  echo "No dataset name supplied"
  exit 1
fi
dataset_name=$2

# Check if ckpt flag is set using getopts
ckpt_flag=0
while getopts ":ckpt:" opt; do
  case $opt in
    ckpt)
      ckpt_flag=1
      ckpt_path=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&3
      ;;
  esac
done

# If checkpoint flag is empty, run chain inference
if [ $ckpt_flag -eq 0 ]
then
  # Running chain inference
  echo "Running chain inference"
  python scripts/chain_inference.py +exp=$1 datamodule.train_dataset=None datamodule.val_dataset=None datamodule.test_dataset.render_root=./RemFX_eval_datasets/ render_files=False num_removed_effects=[${dataset_name:0:1},${dataset_name:2:1}]
  exit 1
fi

# Otherwise run inference on the specified checkpoint
echo "Running monolithic inference on checkpoint $3"
python scripts/test.py +exp=$1 datamodule.train_dataset=None datamodule.val_dataset=None datamodule.test_dataset.render_root=./RemFX_eval_datasets/ datamodule.test_dataset.num_kept_effects="[0,0]" num_removed_effects=[${dataset_name:0:1},${dataset_name:2:1}] effects_to_keep=[] effects_to_remove="[compressor,reverb,chorus,delay,distortion]" render_files=False +ckpt_path=$2



