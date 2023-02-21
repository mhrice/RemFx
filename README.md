
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
2. `python scripts/train.py +exp=umx_distortion`
or
2. `python scripts/train.py +exp=demucs_distortion`
See cfg for more options. Generally they are `+exp={model}_{effect}`
Models and effects detailed below.

To add gpu, add `trainer.accelerator='gpu' trainer.devices=-1` to the command-line

Ex. `python scripts/train.py +exp=umx_distortion trainer.accelerator='gpu' trainer.devices=-1`

### Current Models
- `umx`
- `demucs`

### Current Effects
- `chorus`
- `compressor`
- `distortion`
- `reverb`
- `all` (choose random effect to apply to each file)

## Misc.
By default, files are rendered to `input_dir / processed / train/val/test`.
To skip rendering files (use previously rendered), add `render_files=False` to the command-line
To change the rendered location, add `render_root={path/to/dir}` to the command-line
Test
Experiment dictates data, ckpt dictates model
`python scripts/test.py +exp=umx_distortion.yaml +ckpt_path=test_ckpts/umx_dist.ckpt`