
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
- `num_kept_effects={[min, max]}` range of <b> Kept </b> effects to apply to each file. Inclusive.
- `num_removed_effects={[min, max]}` range of <b> Removed </b> effects to apply to each file. Inclusive.
- `model={model}` architecture to use (see 'Models')
- `effects_to_keep={[effect]}` Effects to apply but not remove (see 'Effects')
- `effects_to_remove={[effect]}` Effects to remove (see 'Effects')
- `accelerator=null/'gpu'` Use GPU (1 device) (default: null)
- `render_files=True/False` Render files. Disable to skip rendering stage (default: True)
- `render_root={path/to/dir}`. Root directory to render files to (default: DATASET_ROOT)

These can also be specified on the command line.
see `cfg/exp/default.yaml` for an example.


## Models
- `umx`
- `demucs`
- `tcn`
- `dcunet`
- `dptnet`

## Effects
- `chorus`
- `compressor`
- `distortion`
- `reverb`
- `delay`

## Chain Inference
`python scripts/chain_inference.py +exp=chain_inference`

## Run inference on directory
Assumes directory is structured as
- root
    - clean
        - file1.wav
        - file2.wav
        - file3.wav
    - effected
        - file1.wav
        - file2.wav
        - file3.wav

Change root path in `shell_vars.sh` and `source shell_vars.sh`

`python scripts/chain_inference.py +exp=chain_inference_custom`



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