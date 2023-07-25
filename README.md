# General Purpose Audio Effect Removal
# About
TBD. Add photo. Add paper link.
# Setup
```
git clone https://github.com/mhrice/RemFx.git
git submodule update --init --recursive
pip install . umx
```
# Usage
## Run RemFX Detect on a single file
```
./download_checkpoints.sh
./remfx_detect.sh wet.wav -o dry.wav
```
## Download the [General Purpose Audio Effect Removal evaluation dataset](https://zenodo.org/record/8183649/)
```
wget https://zenodo.org/record/8183649/files/RemFX_eval_dataset.zip?download=1 -O RemFX_eval_dataset.zip
unzip RemFX_eval_dataset.zip
```

## Download the datasets used in the paper
```
python scripts/download.py vocalset guitarset idmt-smt-guitar idmt-smt-bass idmt-smt-drums
```


## Training
Before training, it is important that you have downloaded the datasets (see above).
This project uses [hydra](https://hydra.cc/) for configuration management. All experiments are defined in `cfg/exp/`. To train with an existing experiment, first run
```
export DATASET_ROOT={path/to/datasets}
```
Then:
```
python scripts/train.py +exp={experiment_name}
```

Here are some selected experiment types from the paper, which use different datasets and configurations. See `cfg/exp/` for a full list of experiments and parameters.

| Experiment Type         | config name  | example          |
| ----------------------- | ------------ | ---------------- |
| Effect-specific         | {effect}     | +exp=chorus      |
| Effect-specific + FXAug | {effect}_aug | +exp=chorus_aug  |
| Monolithic (1 FX)       | 5-5          | +exp=5-1         |
| Monolithic (<=5 FX)     | 5-5          | +exp=5-5         |
| Classifier              | 5-5_cls      | +exp=5-5_cls     |

To change the configuration, simply edit the experiment file, or override the configuration on the command line. A description of some of these variables is in the Misc. section below.
You can also create a custom experiment by creating a new experiment file in `cfg/exp/` and overriding the default parameters in `config.yaml`.

## Evaluate models on the General Purpose Audio Effect Removal evaluation dataset
First download the dataset (see above).
To use the pretrained RemFX model, download the checkpoints
```
./download_checkpoints.sh
```
Then run the evaluation script, select the RemFX configuration, between `remfx_oracle`, `remfx_detect`, and `remfx_all`.
```
./eval.sh remfx_detect
```
To use a custom trained model, first train a model (see Training)
Then run the evaluation script, with config used.
```
./eval.sh {experiment_name}
```

## Checkpoints
Download checkpoints from [here](https://zenodo.org/record/8179396), or see the ./download_checkpoints.sh script.


## Generate datasets used in the paper
```
```
Note that by default, files are rendered to `input_dir / processed / {string_of_effects} / {train|val|test}`.

## Evaluate with a custom directory
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

# Misc.
## Experimental parameters
Some relevant training parameters descriptions
- `num_kept_effects={[min, max]}` range of <b> Kept </b> effects to apply to each file. Inclusive.
- `num_removed_effects={[min, max]}` range of <b> Removed </b> effects to apply to each file. Inclusive.
- `model={model}` architecture to use (see 'Models')
- `effects_to_keep={[effect]}` Effects to apply but not remove (see 'Effects')
- `effects_to_remove={[effect]}` Effects to remove (see 'Effects')
- `accelerator=null/'gpu'` Use GPU (1 device) (default: null)
- `render_files=True/False` Render files. Disable to skip rendering stage (default: True)
- `render_root={path/to/dir}`. Root directory to render files to (default: DATASET_ROOT)

### Effect Removal Models
- `umx`
- `demucs`
- `tcn`
- `dcunet`
- `dptnet`

### Effect Classification Models
- `cls_vggish`
- `cls_panns_pt`
- `cls_wav2vec2`
- `cls_wav2clip`

### Effects
- `chorus`
- `compressor`
- `distortion`
- `reverb`
- `delay`







