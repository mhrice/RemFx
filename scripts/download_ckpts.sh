#! /bin/bash

# make ckpts directory if not exist
mkdir -p ckpts

# download ckpts and save to ckpts directory
wget https://zenodo.org/record/8218621/files/classifier.ckpt?download=1 -O ckpts/classifier.ckpt
wget https://zenodo.org/record/8218621/files/dcunet_chorus_aug.ckpt?download=1 -O ckpts/dcunet_chorus_aug.ckpt
wget https://zenodo.org/record/8218621/files/dcunet_delay_aug.ckpt?download=1 -O ckpts/dcunet_delay_aug.ckpt
wget https://zenodo.org/record/8218621/files/dcunet_reverb_aug.ckpt?download=1 -O ckpts/dcunet_reverb_aug.ckpt
wget https://zenodo.org/record/8218621/files/demucs_compressor_aug.ckpt?download=1 -O ckpts/demucs_compressor_aug.ckpt
wget https://zenodo.org/record/8218621/files/demucs_distortion_aug.ckpt?download=1 -O ckpts/demucs_distortion_aug.ckpt