
## Install Packages
`python3 -m venv env`
`pip install -e .`
`pip install -e umx`

## Download [GuitarFX Dataset] (https://zenodo.org/record/7044411/)
`./download_egfx.sh`

## Train model
1. Change Wandb variables in `shell_vars.sh`
2. `python train.py exp=audio_diffusion`
or
2. `python train.py exp=umx`

To add gpu, add `trainer.accelerator='gpu' trainer.devices=-1` to the command-line
Ex. `python train.py exp=umx trainer.accelerator='gpu' trainer.devices=-1`

