import torch
from torch import Tensor
import pytorch_lightning as pl
from einops import rearrange
import wandb
from audio_diffusion_pytorch import AudioDiffusionModel

import sys

sys.path.append("./umx")
from umx.openunmix.model import OpenUnmix, Separator


SAMPLE_RATE = 22050  # From audio-diffusion-pytorch


class OpenUnmixModel(pl.LightningModule):
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        alpha: float = 0.3,
    ):
        super().__init__()
        self.model = OpenUnmix(
            nb_channels=1,
            nb_bins=n_fft // 2 + 1,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha = alpha
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, Y = self.common_step(batch, batch_idx, mode="val")
        return loss, Y

    def common_step(self, batch, batch_idx, mode: str = "train"):
        x, target, label = batch
        X = spectrogram(x, self.window, self.n_fft, self.hop_length, self.alpha)
        Y = self(X)
        Y_hat = spectrogram(
            target, self.window, self.n_fft, self.hop_length, self.alpha
        )
        loss = torch.nn.functional.mse_loss(Y, Y_hat)
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True)
        return loss, Y

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=1e-4, betas=(0.95, 0.999), eps=1e-6, weight_decay=1e-3
        )

    def on_validation_epoch_start(self):
        self.log_next = True

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.log_next:
            x, target, label = batch
            s = Separator(
                target_models={"other": self.model},
                nb_channels=1,
                sample_rate=SAMPLE_RATE,
                n_fft=self.n_fft,
                n_hop=self.hop_length,
            )
            outputs = s(x).squeeze(1)
            log_wandb_audio_batch(
                id="sample",
                samples=x,
                sampling_rate=SAMPLE_RATE,
                caption=f"Epoch {self.current_epoch}",
            )
            log_wandb_audio_batch(
                id="prediction",
                samples=outputs,
                sampling_rate=SAMPLE_RATE,
                caption=f"Epoch {self.current_epoch}",
            )
            log_wandb_audio_batch(
                id="target",
                samples=target,
                sampling_rate=SAMPLE_RATE,
                caption=f"Epoch {self.current_epoch}",
            )
            self.log_next = False


class DiffusionGenerationModel(pl.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def sample(self, *args, **kwargs) -> Tensor:
        return self.model.sample(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="val")

    def common_step(self, batch, batch_idx, mode: str = "train"):
        x, target, label = batch
        loss = self(x)
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=1e-4, betas=(0.95, 0.999), eps=1e-6, weight_decay=1e-3
        )

    def on_validation_epoch_start(self):
        self.log_next = True

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        x, target, label = batch
        if self.log_next:
            self.log_sample(x)
            self.log_next = False

    @torch.no_grad()
    def log_sample(self, batch, num_steps=10):
        # Get start diffusion noise
        noise = torch.randn(batch.shape, device=self.device)
        sampled = self.sample(noise=noise, num_steps=num_steps)  # Suggested range: 2-50
        log_wandb_audio_batch(
            id="sample",
            samples=sampled,
            sampling_rate=SAMPLE_RATE,
            caption=f"Sampled in {num_steps} steps",
        )


def log_wandb_audio_batch(
    id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = rearrange(samples, "b c t -> b t c")
    for idx in range(num_items):
        wandb.log(
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
