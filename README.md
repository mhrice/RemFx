
## Install Packages
1. `python3 -m venv env`
2. `source env/bin/activate`
3. `pip install -e .`
4. `git submodule update --init --recursive`
5. `pip install -e umx`

## Download [VocalSet Dataset](https://zenodo.org/record/1193957)
1. `wget https://zenodo.org/record/1193957/files/VocalSet.zip?download=1`
2. `mv VocalSet.zip?download=1 VocalSet.zip`
3. `unzip VocalSet.zip`
4. Manually split singers into train, val, test directories

## Train model
1. Change Wandb and data root variables in `shell_vars.sh` and `source shell_vars.sh`
2. `python scripts/train.py model=demucs "effects_to_remove=[distortion]"`


## Models
- `umx`
- `demucs`

## Effects
- `chorus`
- `compressor`
- `distortion`
- `reverb`

## Train CLI Options
- `max_kept_effects={n}` max number of <b> Kept </b> effects to apply to each file (default: 3)
- `model={model}` architecture to use (see 'Models')
- `shuffle_kept_effects=True/False` Shuffle kept effects (default: True)
- `shuffle_removed_effects=True/False` Shuffle Removed effects (default: False)
- `effects_to_use={effect}` Effects to use (see 'Effects') (default: all in the list)
- `effects_to_remove={effect}` Effects to remove (see 'Effects') (default: all in the list)
- `trainer.accelerator='gpu'` : Use GPU (default: None)
- `trainer.devices={n}` Number of GPUs to use (default: 1)
- `render_files=True/False` Render files. Disable to skip rendering stage (default: True)
- `render_root={path/to/dir}`. Root directory to render files to (default: DATASET_ROOT)

Example: `python scripts/train.py model=demucs "effects_to_use=[distortion, reverb]" "effects_to_remove=[distortion]" "max_kept_effects=2" "shuffle_kept_effects=False" "shuffle_removed_effects=True" trainer.accelerator='gpu' trainer.devices=2`


## Misc.
By default, files are rendered to `input_dir / processed / {string_of_effects} / {train|val|test}`.


