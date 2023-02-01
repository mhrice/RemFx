import torch
from torch import Tensor, nn
import pytorch_lightning as pl
from einops import rearrange
import wandb
from audio_diffusion_pytorch import DiffusionModel
from auraloss.time import SISDRLoss
from auraloss.freq import MultiResolutionSTFTLoss, STFTLoss
from torch.nn import L1Loss

from umx.openunmix.model import OpenUnmix, Separator
from torchaudio.models import HDemucs


class RemFXModel(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_eps: float,
        lr_weight_decay: float,
        sample_rate: float,
        network: nn.Module,
    ):
        super().__init__()
        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_eps = lr_eps
        self.lr_weight_decay = lr_weight_decay
        self.sample_rate = sample_rate
        self.model = network
        self.metrics = torch.nn.ModuleDict(
            {
                "SISDR": SISDRLoss(),
                "STFT": STFTLoss(),
                "L1": L1Loss(),
            }
        )

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
        return loss

    def common_step(self, batch, batch_idx, mode: str = "train"):
        loss, output = self.model(batch)
        self.log(f"{mode}_loss", loss)
        x, y, label = batch
        # Metric logging
        for metric in self.metrics:
            self.log(
                f"{mode}_{metric}",
                self.metrics[metric](output, y),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                sync_dist=True,
            )

        return loss

    def on_validation_epoch_start(self):
        self.log_next = True

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.log_next:
            x, target, label = batch
            self.model.eval()
            with torch.no_grad():
                y = self.model.sample(x)

            # Concat samples together for easier viewing in dashboard
            concat_samples = torch.cat([y, x, target], dim=-1)
            log_wandb_audio_batch(
                logger=self.logger,
                id="prediction_input_target",
                samples=concat_samples.cpu(),
                sampling_rate=self.sample_rate,
                caption=f"Epoch {self.current_epoch}",
            )
            self.log_next = False
            self.model.train()


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
        self.loss_fn = MultiResolutionSTFTLoss(
            n_bins=self.num_bins, sample_rate=self.sample_rate
        )

    def forward(self, batch):
        x, target, label = batch
        X = spectrogram(x, self.window, self.n_fft, self.hop_length, self.alpha)
        Y = self.model(X)
        sep_out = self.separator(x).squeeze(1)
        loss = self.loss_fn(sep_out, target)

        return loss, sep_out

    def sample(self, x: Tensor) -> Tensor:
        return self.separator(x).squeeze(1)


class DemucsModel(torch.nn.Module):
    def __init__(self, sample_rate, **kwargs) -> None:
        super().__init__()
        self.model = HDemucs(**kwargs)
        self.num_bins = kwargs["nfft"] // 2 + 1
        self.loss_fn = MultiResolutionSTFTLoss(
            n_bins=self.num_bins, sample_rate=sample_rate
        )

    def forward(self, batch):
        x, target, label = batch
        output = self.model(x).squeeze(1)
        loss = self.loss_fn(output, target)
        return loss, output

    def sample(self, x: Tensor) -> Tensor:
        return self.model(x).squeeze(1)


class DiffusionGenerationModel(nn.Module):
    def __init__(self, n_channels: int = 1):
        super().__init__()
        self.model = DiffusionModel(in_channels=n_channels)

    def forward(self, batch):
        x, target, label = batch
        sampled_out = self.model.sample(x)
        return self.model(x), sampled_out

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
