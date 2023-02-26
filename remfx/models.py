import torch
from torch import Tensor, nn
import pytorch_lightning as pl
from einops import rearrange
import wandb
from audio_diffusion_pytorch import DiffusionModel
from auraloss.time import SISDRLoss
from auraloss.freq import MultiResolutionSTFTLoss
from remfx.utils import FADLoss

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
                "STFT": MultiResolutionSTFTLoss(),
                "FAD": FADLoss(sample_rate=sample_rate),
            }
        )
        # Log first batch metrics input vs output only once
        self.log_train_audio = True

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

    # Add step-based learning rate scheduler
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_lbfgs=False,
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # update learning rate. Reduce by factor of 10 at 80% and 95% of training
        if self.trainer.global_step == 0.8 * self.trainer.max_steps:
            for pg in optimizer.param_groups:
                pg["lr"] = 0.1 * pg["lr"]
        if self.trainer.global_step == 0.95 * self.trainer.max_steps:
            for pg in optimizer.param_groups:
                pg["lr"] = 0.1 * pg["lr"]

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="valid")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="test")
        return loss

    def common_step(self, batch, batch_idx, mode: str = "train"):
        loss, output = self.model(batch)
        self.log(f"{mode}_loss", loss)
        x, y, label = batch
        # Metric logging
        with torch.no_grad():
            for metric in self.metrics:
                # SISDR returns negative values, so negate them
                if metric == "SISDR":
                    negate = -1
                else:
                    negate = 1
                # Only Log FAD on test set
                if metric == "FAD" and mode != "test":
                    continue
                self.log(
                    f"{mode}_{metric}",
                    negate * self.metrics[metric](output, y),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                    sync_dist=True,
                )

        return loss

    def on_train_batch_start(self, batch, batch_idx):
        # Log initial audio
        if self.log_train_audio:
            x, y, label = batch
            # Concat samples together for easier viewing in dashboard
            input_samples = rearrange(x, "b c t -> c (b t)").unsqueeze(0)
            target_samples = rearrange(y, "b c t -> c (b t)").unsqueeze(0)

            log_wandb_audio_batch(
                logger=self.logger,
                id="input_effected_audio",
                samples=input_samples.cpu(),
                sampling_rate=self.sample_rate,
                caption="Training Data",
            )
            log_wandb_audio_batch(
                logger=self.logger,
                id="target_audio",
                samples=target_samples.cpu(),
                sampling_rate=self.sample_rate,
                caption="Target Data",
            )
            self.log_train_audio = False

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        x, target, label = batch
        # Log Input Metrics
        for metric in self.metrics:
            # SISDR returns negative values, so negate them
            if metric == "SISDR":
                negate = -1
            else:
                negate = 1
            # Only Log FAD on test set
            if metric == "FAD":
                continue
            self.log(
                f"Input_{metric}",
                negate * self.metrics[metric](x, target),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                sync_dist=True,
            )
        # Only run on first batch
        if batch_idx == 0:
            self.model.eval()
            with torch.no_grad():
                y = self.model.sample(x)

            # Concat samples together for easier viewing in dashboard
            # 2 seconds of silence between each sample
            silence = torch.zeros_like(x)
            silence = silence[:, : self.sample_rate * 2]

            concat_samples = torch.cat([y, silence, x, silence, target], dim=-1)
            log_wandb_audio_batch(
                logger=self.logger,
                id="prediction_input_target",
                samples=concat_samples.cpu(),
                sampling_rate=self.sample_rate,
                caption=f"Epoch {self.current_epoch}",
            )
            self.model.train()

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self.on_validation_batch_start(batch, batch_idx, dataloader_idx)
        # Log FAD
        x, target, label = batch
        self.log(
            "Input_FAD",
            self.metrics["FAD"](x, target),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )


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
        self.mrstftloss = MultiResolutionSTFTLoss(
            n_bins=self.num_bins, sample_rate=self.sample_rate
        )
        self.l1loss = torch.nn.L1Loss()

    def forward(self, batch):
        x, target, label = batch
        X = spectrogram(x, self.window, self.n_fft, self.hop_length, self.alpha)
        Y = self.model(X)
        sep_out = self.separator(x).squeeze(1)
        loss = self.mrstftloss(sep_out, target) + self.l1loss(sep_out, target) * 100

        return loss, sep_out

    def sample(self, x: Tensor) -> Tensor:
        return self.separator(x).squeeze(1)


class DemucsModel(torch.nn.Module):
    def __init__(self, sample_rate, **kwargs) -> None:
        super().__init__()
        self.model = HDemucs(**kwargs)
        self.num_bins = kwargs["nfft"] // 2 + 1
        self.mrstftloss = MultiResolutionSTFTLoss(
            n_bins=self.num_bins, sample_rate=sample_rate
        )
        self.l1loss = torch.nn.L1Loss()

    def forward(self, batch):
        x, target, label = batch
        output = self.model(x).squeeze(1)
        loss = self.mrstftloss(output, target) + self.l1loss(output, target) * 100
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
