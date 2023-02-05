
## Install Packages
1. `python3 -m venv env`
2. `source env/bin/activate`
3. `pip install -e .`
4. `git submodule update --init --recursive`
5. `pip install -e umx`

## Download [GuitarFX Dataset](https://zenodo.org/record/7044411/)
`./scripts/download_egfx.sh`

## Train model
1. Change Wandb variables in `shell_vars.sh`
2. `python scripts/train.py exp=audio_diffusion`
or
2. `python scripts/train.py exp=umx`

To add gpu, add `trainer.accelerator='gpu' trainer.devices=-1` to the command-line

Ex. `python train.py exp=umx trainer.accelerator='gpu' trainer.devices=-1`

### Effects
Default effect is RAT (distortion). Effect choices:
- BluesDriver
- Clean
- Flanger
- Phaser
- RAT
- Sweep Echo
- TubeScreamer
- Chorus
- Digital Delay
- Hall Reverb
- Plate Reverb
- Spring Reverb
- TapeEcho

Change effect by adding `+datamodule.dataset.effect_types=["{Effect}"]` to the command-line