import wandb
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import Tensor, nn
from einops import rearrange
from auraloss.time import SISDRLoss
from torchaudio.models import HDemucs
from auraloss.freq import MultiResolutionSTFTLoss
from audio_diffusion_pytorch import DiffusionModel
from umx.openunmix.model import OpenUnmix, Separator

from remfx.utils import FADLoss


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
        on_tpu,
        using_lbfgs,
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


# adapted from https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py


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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class Cnn14(nn.Module):
    def __init__(
        self,
        num_classes: int,
        sample_rate: float,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_fft = n_fft
        self.hop_length = hop_length

        window = torch.hann(n_fft)
        self.register_buffer("window", window)

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate,
            n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, num_classes, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x: torch.Tensor):
        """
        Input: (batch_size, data_length)"""

        x = self.melspec(x)
        x = self.bn0(x)

        if self.training:
            pass
            # x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output


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
        x, y, label = batch
        pred_label = self.network(x)
        loss = torch.nn.functional.cross_entropy(pred_label, label)
        self.log(
            f"{mode}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def train_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer
