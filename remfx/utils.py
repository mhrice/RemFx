import logging
from typing import List, Tuple
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only
from frechet_audio_distance import FrechetAudioDistance
import numpy as np
import torch
import torchaudio


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
    chunk_size_in_samples = chunk_size * sr
    chunk_starts = torch.arange(0, audio.shape[-1], chunk_size_in_samples)
    for start in chunk_starts:
        if start + chunk_size_in_samples > audio.shape[-1]:
            break
        chunks.append(audio[:, start : start + chunk_size_in_samples])
    return chunks, sr
