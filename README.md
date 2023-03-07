
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

## Train Main CLI Options
- `max_kept_effects={n}` max number of <b> Kept </b> effects to apply to each file (default: 3)
- `max_removed_effects={n}` max number of <b> Removed </b> effects to apply to each file (default: 4)
- `model={model}` architecture to use (see 'Models')
- `shuffle_kept_effects=True/False` Shuffle kept effects (default: True)
- `shuffle_removed_effects=True/False` Shuffle removed effects (default: False)
- `effects_to_use={effect}` Effects to use (see 'Effects') (default: all in the list)
- `effects_to_remove={effect}` Effects to remove (see 'Effects') (default: all in the list)
- `accelerator=null/gpu` Use GPU (1 device) (default: False)
- `render_files=True/False` Render files. Disable to skip rendering stage (default: True)
- `render_root={path/to/dir}`. Root directory to render files to (default: DATASET_ROOT)

Example: `python scripts/train.py model=demucs "effects_to_use=[distortion, reverb]" "effects_to_remove=[distortion]" max_kept_effects=2 max_removed_effects=4 shuffle_kept_effects=False shuffle_removed_effects=True accelerator='gpu' render_root='/home/username/datasets/vocalset'`

See `cfg/config.yaml` for more options that can be specified on the command line.

## Misc.
By default, files are rendered to `input_dir / processed / {string_of_effects} / {train|val|test}`.


