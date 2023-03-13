import logging
from typing import List, Tuple
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only
from frechet_audio_distance import FrechetAudioDistance
import numpy as np
import torch
import torchaudio
from torch import Tensor, nn
import wandb
from einops import rearrange
from torch._six import container_abcs


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: pl.loggers.logger.Logger,
) -> None:
    """Controls which config parts are saved by Lightning loggers.
    Additionaly saves:
    - number of model parameters
    """

    if not trainer.logger:
        return

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    logger.experiment.config.update(hparams)


class FADLoss(torch.nn.Module):
    def __init__(self, sample_rate: float):
        super().__init__()
        self.fad = FrechetAudioDistance(
            use_pca=False, use_activation=False, verbose=False
        )
        self.fad.model = self.fad.model.to("cpu")
        self.sr = sample_rate

    def forward(self, audio_background, audio_eval):
        embds_background = []
        embds_eval = []
        for sample in audio_background:
            embd = self.fad.model.forward(sample.T.cpu().detach().numpy(), self.sr)
            embds_background.append(embd.cpu().detach().numpy())
        for sample in audio_eval:
            embd = self.fad.model.forward(sample.T.cpu().detach().numpy(), self.sr)
            embds_eval.append(embd.cpu().detach().numpy())
        embds_background = np.concatenate(embds_background, axis=0)
        embds_eval = np.concatenate(embds_eval, axis=0)
        mu_background, sigma_background = self.fad.calculate_embd_statistics(
            embds_background
        )
        mu_eval, sigma_eval = self.fad.calculate_embd_statistics(embds_eval)

        fad_score = self.fad.calculate_frechet_distance(
            mu_background, sigma_background, mu_eval, sigma_eval
        )
        return fad_score


def create_random_chunks(
    audio_file: str, chunk_size: int, num_chunks: int
) -> Tuple[List[Tuple[int, int]], int]:
    """Create num_chunks random chunks of size chunk_size (seconds)
    from an audio file.
    Return sample_index of start of each chunk and original sr
    """
    audio, sr = torchaudio.load(audio_file)
    chunk_size_in_samples = chunk_size * sr
    if chunk_size_in_samples >= audio.shape[-1]:
        chunk_size_in_samples = audio.shape[-1] - 1
    chunks = []
    for i in range(num_chunks):
        start = torch.randint(0, audio.shape[-1] - chunk_size_in_samples, (1,)).item()
        chunks.append(start)
    return chunks, sr


def create_sequential_chunks(
    audio_file: str, chunk_size: int
) -> Tuple[List[Tuple[int, int]], int]:
    """Create sequential chunks of size chunk_size (seconds) from an audio file.
    Return sample_index of start of each chunk and original sr
    """
    chunks = []
    audio, sr = torchaudio.load(audio_file)
    chunk_starts = torch.arange(0, audio.shape[-1], chunk_size)
    for start in chunk_starts:
        if start + chunk_size > audio.shape[-1]:
            break
        chunks.append(audio[:, start : start + chunk_size])
    return chunks, sr


def log_wandb_audio_batch(
    logger: pl.loggers.WandbLogger,
    id: str,
    samples: Tensor,
    sampling_rate: int,
    caption: str = "",
    max_items: int = 10,
):
    num_items = samples.shape[0]
    samples = rearrange(samples, "b c t -> b t c")
    for idx in range(num_items):
        if idx >= max_items:
            break
        logger.experiment.log(
            {
                f"{id}_{idx}": wandb.Audio(
                    samples[idx].cpu().numpy(),
                    caption=caption,
                    sample_rate=sampling_rate,
                )
            }
        )


def spectrogram(
    x: torch.Tensor,
    window: torch.Tensor,
    n_fft: int,
    hop_length: int,
    alpha: float,
) -> torch.Tensor:
    bs, chs, samp = x.size()
    x = x.view(bs * chs, -1)  # move channels onto batch dim

    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )

    # move channels back
    X = X.view(bs, chs, X.shape[-2], X.shape[-1])

    return torch.pow(X.abs() + 1e-8, alpha)


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def _ntuple(n: int):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple([x] * n)

    return parse


single = _ntuple(1)


def concat_complex(a: torch.tensor, b: torch.tensor, dim: int = 1) -> torch.tensor:
    """
    Concatenate two complex tensors in same dimension concept
    :param a: complex tensor
    :param b: another complex tensor
    :param dim: target dimension
    :return: concatenated tensor
    """
    a_real, a_img = a.chunk(2, dim)
    b_real, b_img = b.chunk(2, dim)
    return torch.cat([a_real, b_real, a_img, b_img], dim=dim)
