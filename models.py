import torch
from torch import Tensor, nn
import pytorch_lightning as pl
from einops import rearrange
import wandb
from audio_diffusion_pytorch import AudioDiffusionModel
import auraloss

import sys

sys.path.append("./umx")
from umx.openunmix.model import OpenUnmix, Separator


SAMPLE_RATE = 22050  # From audio-diffusion-pytorch


class RemFXModel(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_eps: float,
        lr_weight_decay: float,
        network: nn.Module,
    ):
        super().__init__()
        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_eps = lr_eps
        self.lr_weight_decay = lr_weight_decay
        self.model = network

    @property
    def device(self):
        return next(self.model.parameters()).device

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="valid")

    def common_step(self, batch, batch_idx, mode: str = "train"):
        loss = self.model(batch)
        self.log(f"{mode}_loss", loss)
        return loss

    def on_validation_epoch_start(self):
        self.log_next = True

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.log_next:
            x, target, label = batch
            y = self.model.sample(x)
            log_wandb_audio_batch(
                logger=self.logger,
                id="sample",
                samples=x.cpu(),
                sampling_rate=SAMPLE_RATE,
                caption=f"Epoch {self.current_epoch}",
            )
            log_wandb_audio_batch(
                logger=self.logger,
                id="prediction",
                samples=y.cpu(),
                sampling_rate=SAMPLE_RATE,
                caption=f"Epoch {self.current_epoch}",
            )
            log_wandb_audio_batch(
                logger=self.logger,
                id="target",
                samples=target.cpu(),
                sampling_rate=SAMPLE_RATE,
                caption=f"Epoch {self.current_epoch}",
            )
            self.log_next = False


class OpenUnmixModel(torch.nn.Module):
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_channels: int = 1,
        alpha: float = 0.3,
        sample_rate: int = 22050,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha = alpha
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

        self.num_bins = self.n_fft // 2 + 1
        self.sample_rate = sample_rate
        self.model = OpenUnmix(
            nb_channels=self.n_channels,
            nb_bins=self.num_bins,
        )
        self.separator = Separator(
            target_models={"other": self.model},
            nb_channels=self.n_channels,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_hop=self.hop_length,
        )
        self.loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
            n_bins=self.num_bins, sample_rate=self.sample_rate
        )

    def forward(self, batch):
        x, target, label = batch
        X = spectrogram(x, self.window, self.n_fft, self.hop_length, self.alpha)
        Y = self.model(X)
        sep_out = self.separator(x).squeeze(1)
        loss = self.loss_fn(sep_out, target)

        return loss

    def sample(self, x: Tensor) -> Tensor:
        return self.separator(x).squeeze(1)


class DiffusionGenerationModel(nn.Module):
    def __init__(self, n_channels: int = 1):
        super().__init__()
        self.model = AudioDiffusionModel(in_channels=n_channels)

    def forward(self, batch):
        x, target, label = batch
        return self.model(x)

    def sample(self, x: Tensor, num_steps: int = 10) -> Tensor:
        noise = torch.randn(x.shape).to(x)
        return self.model.sample(noise, num_steps=num_steps)


def log_wandb_audio_batch(
    logger: pl.loggers.WandbLogger,
    id: str,
    samples: Tensor,
    sampling_rate: int,
    caption: str = "",
):
    num_items = samples.shape[0]
    samples = rearrange(samples, "b c t -> b t c")
    for idx in range(num_items):
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
