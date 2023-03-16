
# Setup

## Install Packages
1. `python3 -m venv env`
2. `source env/bin/activate`
3. `pip install -e .`
4. `git submodule update --init --recursive`
5. `pip install -e umx`

## Download [VocalSet Dataset](https://zenodo.org/record/1193957)
1. `wget https://zenodo.org/record/1442513/files/VocalSet1-2.zip?download=1`
2. `mv VocalSet.zip?download=1 VocalSet.zip`
3. `unzip VocalSet.zip`

# Training
## Steps
1. Change Wandb and data root variables in `shell_vars.sh` and `source shell_vars.sh`
2. `python scripts/train.py +exp=default`

## Experiments
Training parameters can be configured in `cfg/exp/default.yaml`. Here are some descriptions
- `max_kept_effects={n}` max number of <b> Kept </b> effects to apply to each file. Set to -1 to always use all effects (default: -1)
- `max_removed_effects={n}` max number of <b> Removed </b> effects to apply to each file. Set to -1 to always use all effects (default: -1)
- `model={model}` architecture to use (see 'Models')
- `effects_to_use={effect}` Effects to use (see 'Effects') (default: all in the list)
- `effects_to_remove={effect}` Effects to remove (see 'Effects') (default: all in the list)
- `accelerator=null/'gpu'` Use GPU (1 device) (default: null)
- `render_files=True/False` Render files. Disable to skip rendering stage (default: True)
- `render_root={path/to/dir}`. Root directory to render files to (default: DATASET_ROOT)

Note that "kept effects" are calculated from the difference between `effects_to_use` and `effects_to_remove`.

These can also be specified on the command line.
Example: `python scripts/train.py model=demucs "effects_to_use=[distortion, reverb, chorus]" "effects_to_remove=[distortion]" max_kept_effects=2 max_removed_effects=4 shuffle_kept_effects=False shuffle_removed_effects=True accelerator='gpu' render_root=/scratch/VocalSet'`

Printout:
```
Effect Summary:
Apply kept effects: ['chorus', 'reverb'] (Up to 2, chosen in order) -> Dry
Apply remove effects: ['distortion'] (Up to 4, chosen randomly) -> Wet
```

## Models
- `umx`
- `demucs`

## Effects
- `chorus`
- `compressor`
- `distortion`
- `reverb`

## Misc.
By default, files are rendered to `input_dir / processed / {string_of_effects} / {train|val|test}`.


Download datasets:

```
python scripts/download.py vocalset guitarset idmt-smt-guitar idmt-smt-bass idmt-smt-drums
```

To run audio effects classifiction:
```
python scripts/train.py model=classifier "effects_to_use=[compressor, distortion, reverb, chorus, delay]" "effects_to_remove=[]" max_kept_effects=5 max_removed_effects=0 shuffle_kept_effects=True shuffle_removed_effects=True accelerator='gpu' render_root=/scratch/RemFX render_files=True
```