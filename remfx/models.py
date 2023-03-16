import torch
import torchmetrics
import pytorch_lightning as pl
from torch import Tensor, nn
from torch.nn import functional as F
from torchaudio.models import HDemucs
from audio_diffusion_pytorch import DiffusionModel
from auraloss.time import SISDRLoss
from auraloss.freq import MultiResolutionSTFTLoss
from umx.openunmix.model import OpenUnmix, Separator

from remfx.utils import FADLoss, spectrogram
from remfx.dptnet import DPTNet_base
from remfx.dcunet import RefineSpectrogramUnet
from remfx.tcn import TCN
from remfx.utils import causal_crop


class RemFX(pl.LightningModule):
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
        self.metrics = nn.ModuleDict(
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
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            [0.8 * self.trainer.max_steps, 0.95 * self.trainer.max_steps],
            gamma=0.1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="valid")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="test")

    def common_step(self, batch, batch_idx, mode: str = "train"):
        x, y, _, _ = batch  # x, y = (B, C, T), (B, C, T)

        loss, output = self.model((x, y))
        self.log(f"{mode}_loss", loss)
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


class OpenUnmixModel(nn.Module):
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
        self.l1loss = nn.L1Loss()

    def forward(self, batch):
        x, target = batch
        X = spectrogram(x, self.window, self.n_fft, self.hop_length, self.alpha)
        Y = self.model(X)
        sep_out = self.separator(x).squeeze(1)
        loss = self.mrstftloss(sep_out, target) + self.l1loss(sep_out, target) * 100

        return loss, sep_out

    def sample(self, x: Tensor) -> Tensor:
        return self.separator(x).squeeze(1)


class DemucsModel(nn.Module):
    def __init__(self, sample_rate, **kwargs) -> None:
        super().__init__()
        self.model = HDemucs(**kwargs)
        self.num_bins = kwargs["nfft"] // 2 + 1
        self.mrstftloss = MultiResolutionSTFTLoss(
            n_bins=self.num_bins, sample_rate=sample_rate
        )
        self.l1loss = nn.L1Loss()

    def forward(self, batch):
        x, target = batch
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
        x, target = batch
        sampled_out = self.model.sample(x)
        return self.model(x), sampled_out

    def sample(self, x: Tensor, num_steps: int = 10) -> Tensor:
        noise = torch.randn(x.shape).to(x)
        return self.model.sample(noise, num_steps=num_steps)


class DPTNetModel(nn.Module):
    def __init__(self, sample_rate, num_bins, **kwargs):
        super().__init__()
        self.model = DPTNet_base(**kwargs)
        self.num_bins = num_bins
        self.mrstftloss = MultiResolutionSTFTLoss(
            n_bins=self.num_bins, sample_rate=sample_rate
        )
        self.l1loss = nn.L1Loss()

    def forward(self, batch):
        x, target = batch
        output = self.model(x.squeeze(1))
        loss = self.mrstftloss(output, target) + self.l1loss(output, target) * 100
        return loss, output

    def sample(self, x: Tensor) -> Tensor:
        return self.model(x.squeeze(1))


class DCUNetModel(nn.Module):
    def __init__(self, sample_rate, num_bins, **kwargs):
        super().__init__()
        self.model = RefineSpectrogramUnet(**kwargs)
        self.mrstftloss = MultiResolutionSTFTLoss(
            n_bins=num_bins, sample_rate=sample_rate
        )
        self.l1loss = nn.L1Loss()

    def forward(self, batch):
        x, target = batch
        output = self.model(x.squeeze(1)).unsqueeze(1)  # B x 1 x T
        # Crop target to match output
        if output.shape[-1] < target.shape[-1]:
            target = causal_crop(target, output.shape[-1])
        loss = self.mrstftloss(output, target) + self.l1loss(output, target) * 100
        return loss, output

    def sample(self, x: Tensor) -> Tensor:
        output = self.model(x.squeeze(1)).unsqueeze(1)  # B x 1 x T
        return output


class TCNModel(nn.Module):
    def __init__(self, sample_rate, num_bins, **kwargs):
        super().__init__()
        self.model = TCN(**kwargs)
        self.mrstftloss = MultiResolutionSTFTLoss(
            n_bins=num_bins, sample_rate=sample_rate
        )
        self.l1loss = nn.L1Loss()

    def forward(self, batch):
        x, target = batch
        output = self.model(x)  # B x 1 x T
        # Crop target to match output
        if output.shape[-1] < target.shape[-1]:
            target = causal_crop(target, output.shape[-1])
        loss = self.mrstftloss(output, target) + self.l1loss(output, target) * 100
        return loss, output

    def sample(self, x: Tensor) -> Tensor:
        output = self.model(x)  # B x 1 x T
        return output


class FXClassifier(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_weight_decay: float,
        sample_rate: float,
        network: nn.Module,
    ):
        super().__init__()
        self.lr = lr
        self.lr_weight_decay = lr_weight_decay
        self.sample_rate = sample_rate
        self.network = network

    def forward(self, x: torch.Tensor):
        return self.network(x)

    def common_step(self, batch, batch_idx, mode: str = "train"):
        x, y, dry_label, wet_label = batch
        pred_label = self.network(x)
        loss = nn.functional.cross_entropy(pred_label, dry_label)
        self.log(
            f"{mode}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            f"{mode}_mAP",
            torchmetrics.functional.retrieval_average_precision(
                pred_label, dry_label.long()
            ),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="valid")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.lr,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer
