#! /bin/bash

# Example usage:
# ./eval.sh remfx_detect 

# Check if first argument is empty
if [ -z "$1" ]
then
  echo "No experiment name or config path supplied"
  exit 1
fi

python scripts/chain_inference.py +exp=$1 datamodule.train_dataset=None datamodule.val_dataset=None datamodule.test_dataset.render_root=./RemFX_eval_dataset/ render_files=False



