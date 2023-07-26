# General Purpose Audio Effect Removal
Removing multiple audio effects from multiple sources using compositional audio effect removal and source separation and speech enhancement models.

This repo contains the code for the paper [General Purpose Audio Effect Removal](https://arxiv.org/abs/2110.00484). (Todo: Link broken, Add video, Add img, citation)


# Setup
```
git clone https://github.com/mhrice/RemFx.git
cd RemFx
git submodule update --init --recursive
pip install -e . ./umx
```
# Usage
This repo can be used for many different tasks. Here are some examples.
## Run RemFX Detect on a single file - []
First, need to download the checkpoints from [zenodo](https://zenodo.org/record/8179396)
```
./download_checkpoints.sh
./remfx_detect.sh wet.wav -o dry.wav
```
## Download the [General Purpose Audio Effect Removal evaluation datasets](https://zenodo.org/record/8183649/) - [x]
```
./download_eval_datasets.sh
```

## Download the starter datasets - [x]
```
python scripts/download.py vocalset guitarset dsd100 idmt-smt-drums
```
By default, the starter datasets are downloaded to `./data/remfx-data`. To change this, pass `--output_dir={path/to/datasets}` to `download.py`

Then set the dataset root :
```
export DATASET_ROOT={path/to/datasets}
```

## Training - [x]
Before training, it is important that you have downloaded the starter datasets (see above) and set DATASET_ROOT.
This project uses the [pytorch-lightning](https://www.pytorchlightning.ai/index.html) framework and [hydra](https://hydra.cc/) for configuration management. All experiments are defined in `cfg/exp/`. To train with an existing experiment run
```
python scripts/train.py +exp={experiment_name}
```

Here are some selected experiment types from the paper, which use different datasets and configurations. See `cfg/exp/` for a full list of experiments and parameters.

| Experiment Type         | Config Name  | Example           |
| ----------------------- | ------------ | ----------------- |
| Effect-specific         | {effect}     | +exp=chorus       |
| Effect-specific + FXAug | {effect}_aug | +exp=chorus_aug   |
| Monolithic (1 FX)       | 5-1          | +exp=5-1          |
| Monolithic (<=5 FX)     | 5-5_full     | +exp=5-5_full     |
| Classifier              | 5-5_full_cls | +exp=5-5_full_cls |

To change the configuration, simply edit the experiment file, or override the configuration on the command line. A description of some of these variables is in the Misc. section below.
You can also create a custom experiment by creating a new experiment file in `cfg/exp/` and overriding the default parameters in `config.yaml`.

At the end of training, the train script will automatically evaluate the test set using the best checkpoint (by validation loss). If epoch 0 is not finished, it will throw an error. To evaluate a specific checkpoint, run

```
python scripts/test.py +exp={experiment_name} +ckpt_path="{path/to/checkpoint}" render_files=False
```

The checkpoints will be saved in `./logs/ckpts/{timestamp}`
Metrics and hyperparams will be logged in `./lightning_logs/{timestamp}`

By default, the dataset needed for the experiment is generated before training.
If you have generated the dataset separately (see Generate datasets used in the paper), be sure to set `render_files=False` in the config or command-line, and set `render_root={path/to/dataset}` if it is in a custom location.

Also note that the training assumes you have a GPU. To train on CPU, set `accelerator=null` in the config or command-line.

## Evaluate models on the General Purpose Audio Effect Removal evaluation datasets (Table 4 from the paper) - []
First download the General Purpose Audio Effect Removal evaluation datasets (see above).
To use the pretrained RemFX model, download the checkpoints
```
./download_checkpoints.sh
```
Then run the evaluation script, select the RemFX configuration, between `remfx_oracle`, `remfx_detect`, and `remfx_all`. Then select N, the number of effects to remove.
```
./eval.sh remfx_detect 0-0
./eval.sh remfx_detect 1-1
./eval.sh remfx_detect 2-2
./eval.sh remfx_detect 3-3
./eval.sh remfx_detect 4-4
./eval.sh remfx_detect 5-5

```
To eval a custom monolithic model, first train a model (see Training)
Then run the evaluation script, with the config used and checkpoint_path.
```
./eval.sh distortion_aug 0-0 -ckpt "logs/ckpts/2023-07-26-10-10-27/epoch\=05-valid_loss\=8.623.ckpt"
./eval.sh distortion_aug 1-1 -ckpt "logs/ckpts/2023-07-26-10-10-27/epoch\=05-valid_loss\=8.623.ckpt"
./eval.sh distortion_aug 2-2 -ckpt "logs/ckpts/2023-07-26-10-10-27/epoch\=05-valid_loss\=8.623.ckpt"
./eval.sh distortion_aug 3-3 -ckpt "logs/ckpts/2023-07-26-10-10-27/epoch\=05-valid_loss\=8.623.ckpt"
./eval.sh distortion_aug 4-4 -ckpt "logs/ckpts/2023-07-26-10-10-27/epoch\=05-valid_loss\=8.623.ckpt"
./eval.sh distortion_aug 5-5 -ckpt "logs/ckpts/2023-07-26-10-10-27/epoch\=05-valid_loss\=8.623.ckpt"
```

To eval a custom effect-specific model as part of the inference chain, first train a model (see Training), then edit `cfg/exp/remfx_{desired_configuration}.yaml` -> ckpts -> {effect}.
Then run the evaluation script.
```
./eval.sh remfx_detect 0-0
```

The script assumes that RemFX_eval_datasets is in the top-level directory.
Metrics and hyperparams will be logged in `./lightning_logs/{timestamp}`

## Generate other datasets - [x]
The datasets used in the experiments are customly generated from the starter datasets. In short, for each training/val/testing example, we select a random 5.5s segment from one of the starter datasets and apply a random number of effects to it. The number of effects applied is controlled by the `num_kept_effects` and `num_removed_effects` parameters. The effects applied are controlled by the `effects_to_keep` and `effects_to_remove` parameters.

Before generating datasets, it is important that you have downloaded the starter datasets (see above) and set DATASET_ROOT.

To generate one of the datasets used in the paper, use of the experiments defined in `cfg/exp/`.
For example, to generate the `chorus` FXAug dataset, which includes files with 5 possible effects, up to 4 kept effects (distortion, reverb, compression, delay), and 1 removed effects (chorus), run
```
python scripts/generate_dataset.py +exp=chorus_aug
```

See the Misc. section below for a description of the parameters.
By default, files are rendered to `{render_root} / processed / {string_of_effects} / {train|val|test}`.

If training, this process will be done automatically at the start of training. To disable this, set `render_files=False` in the config or command-line, and set `render_root={path/to/dataset}` if it is in a custom location.

# Misc.
## Experimental parameters
Some relevant dataset/training parameters descriptions
- `num_kept_effects={[min, max]}` range of <b> Kept </b> effects to apply to each file. Inclusive.
- `num_removed_effects={[min, max]}` range of <b> Removed </b> effects to apply to each file. Inclusive.
- `model={model}` architecture to use (see 'Effect Removal Models/Effect Classification Models')
- `effects_to_keep={[effect]}` Effects to apply but not remove (see 'Effects'). Used for FXAug.
- `effects_to_remove={[effect]}` Effects to remove (see 'Effects')
- `accelerator=null/'gpu'` Use GPU (1 device) (default: null)
- `render_files=True/False` Render files. Disable to skip rendering stage (default: True)
- `render_root={path/to/dir}`. Root directory to render files to (default: ./data)
- `datamodule.train_batch_size={batch_size}`. Change batch size (default: varies)

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

# DO WE NEED THIS?
## Evaluate RemFXwith a custom directory - []
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

First set the dataset root:
```
export DATASET_ROOT={path/to/datasets}
```

Then run
```
python scripts/chain_inference.py +exp=chain_inference_custom
```
