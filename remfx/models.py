import torch
import numpy as np
import torchmetrics
import pytorch_lightning as pl
from torch import Tensor, nn
from torchaudio.models import HDemucs
from audio_diffusion_pytorch import DiffusionModel
from auraloss.time import SISDRLoss
from auraloss.freq import MultiResolutionSTFTLoss
from umx.openunmix.model import OpenUnmix, Separator

from remfx.utils import FADLoss, spectrogram
from remfx.tcn import TCN
from remfx.utils import causal_crop
from remfx.callbacks import log_wandb_audio_batch
from einops import rearrange
from remfx import effects
import asteroid
import random

ALL_EFFECTS = effects.Pedalboard_Effects


class RemFXChainInference(pl.LightningModule):
    def __init__(
        self,
        models,
        sample_rate,
        num_bins,
        effect_order,
        classifier=None,
        shuffle_effect_order=False,
        use_all_effect_models=False,
    ):
        super().__init__()
        self.model = models
        self.mrstftloss = MultiResolutionSTFTLoss(
            n_bins=num_bins, sample_rate=sample_rate
        )
        self.l1loss = nn.L1Loss()
        self.metrics = nn.ModuleDict(
            {
                "SISDR": SISDRLoss(),
                "STFT": MultiResolutionSTFTLoss(),
            }
        )
        self.sample_rate = sample_rate
        self.effect_order = effect_order
        self.classifier = classifier
        self.shuffle_effect_order = shuffle_effect_order
        self.output_str = "IN_SISDR,OUT_SISDR,IN_STFT,OUT_STFT\n"
        self.use_all_effect_models = use_all_effect_models

    def forward(self, batch, batch_idx, order=None):
        x, y, _, rem_fx_labels = batch
        # Use chain of effects defined in config
        if order:
            effects_order = order
        else:
            effects_order = self.effect_order

        # Use classifier labels
        if self.classifier:
            threshold = 0.5
            with torch.no_grad():
                labels = torch.sigmoid(self.classifier(x))
                rem_fx_labels = torch.where(labels > threshold, 1.0, 0.0)
        if self.use_all_effect_models:
            effects_present = [
                [ALL_EFFECTS[i] for i, effect in enumerate(effect_label)]
                for effect_label in rem_fx_labels
            ]
        else:
            effects_present = [
                [
                    ALL_EFFECTS[i]
                    for i, effect in enumerate(effect_label)
                    if effect == 1.0
                ]
                for effect_label in rem_fx_labels
            ]

        output = []
        # input_samples = rearrange(x, "b c t -> c (b t)").unsqueeze(0)
        # target_samples = rearrange(y, "b c t -> c (b t)").unsqueeze(0)

        # log_wandb_audio_batch(
        #     logger=self.logger,
        #     id="input_effected_audio",
        #     samples=input_samples.cpu(),
        #     sampling_rate=self.sample_rate,
        #     caption="Input Data",
        # )
        # log_wandb_audio_batch(
        #     logger=self.logger,
        #     id="target_audio",
        #     samples=target_samples.cpu(),
        #     sampling_rate=self.sample_rate,
        #     caption="Target Data",
        # )
        with torch.no_grad():
            for i, (elem, effects_list) in enumerate(zip(x, effects_present)):
                elem = elem.unsqueeze(0)  # Add batch dim
                # Get the correct effect by search for names in effects_order
                effect_list_names = [effect.__name__ for effect in effects_list]
                effects = [
                    effect for effect in effects_order if effect in effect_list_names
                ]

                # log_wandb_audio_batch(
                #     logger=self.logger,
                #     id=f"{i}_Before",
                #     samples=elem.cpu(),
                #     sampling_rate=self.sample_rate,
                #     caption=effects,
                # )
                for effect in effects:
                    # Sample the model
                    elem = self.model[effect].model.sample(elem)
                #     log_wandb_audio_batch(
                #         logger=self.logger,
                #         id=f"{i}_{effect}",
                #         samples=elem.cpu(),
                #         sampling_rate=self.sample_rate,
                #         caption=effects,
                #     )
                # log_wandb_audio_batch(
                #     logger=self.logger,
                #     id=f"{i}_After",
                #     samples=elem.cpu(),
                #     sampling_rate=self.sample_rate,
                #     caption=effects,
                # )
                output.append(elem.squeeze(0))
        output = torch.stack(output)

        # log_wandb_audio_batch(
        #     logger=self.logger,
        #     id="output_audio",
        #     samples=output_samples.cpu(),
        #     sampling_rate=self.sample_rate,
        #     caption="Output Data",
        # )
        loss = self.mrstftloss(output, y) + self.l1loss(output, y) * 100
        return loss, output

    def test_step(self, batch, batch_idx):
        x, y, _, _ = batch  # x, y = (B, C, T), (B, C, T)
        if self.shuffle_effect_order:
            # Random order
            random.shuffle(self.effect_order)
        loss, output = self.forward(batch, batch_idx, order=self.effect_order)
        # Crop target to match output
        if output.shape[-1] < y.shape[-1]:
            y = causal_crop(y, output.shape[-1])
        self.log("test_loss", loss)
        # Metric logging
        with torch.no_grad():
            for metric in self.metrics:
                # SISDR returns negative values, so negate them
                if metric == "SISDR":
                    negate = -1
                else:
                    negate = 1
                self.log(
                    f"test_{metric}",  # + "".join(self.effect_order).replace("RandomPedalboard", ""),
                    negate * self.metrics[metric](output, y),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"Input_{metric}",
                    negate * self.metrics[metric](x, y),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                    sync_dist=True,
                )
                # print(f"Input_{metric}", negate * self.metrics[metric](x, y))
                # print(f"test_{metric}", negate * self.metrics[metric](output, y))
                self.output_str += f"{negate * self.metrics[metric](x, y).item():.4f},{negate * self.metrics[metric](output, y).item():.4f},"
            self.output_str += "\n"
        return loss

    def on_test_end(self) -> None:
        with open("output.csv", "w") as f:
            f.write(self.output_str)

    def sample(self, batch):
        return self.forward(batch, 0)[1]


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
            }
        )
        # Log first batch metrics input vs output only once
        self.log_train_audio = True
        self.output_str = "IN_SISDR,OUT_SISDR,IN_STFT,OUT_STFT\n"

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
        # Crop target to match output
        target = y
        if output.shape[-1] < y.shape[-1]:
            target = causal_crop(y, output.shape[-1])
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
                    negate * self.metrics[metric](output, target),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                    sync_dist=True,
                )

                self.log(
                    f"Input_{metric}",
                    negate * self.metrics[metric](x, y),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                    sync_dist=True,
                )
                # print(f"Input_{metric}", negate * self.metrics[metric](x, y))
                # print(f"test_{metric}", negate * self.metrics[metric](output, y))
                self.output_str += f"{negate * self.metrics[metric](x, y).item():.4f},{negate * self.metrics[metric](output, y).item():.4f},"
            self.output_str += "\n"
        return loss

    def on_test_end(self) -> None:
        with open("output.csv", "w") as f:
            f.write(self.output_str)


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
        self.model = asteroid.models.dptnet.DPTNet(**kwargs)
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
        self.model = asteroid.models.DCUNet(**kwargs)
        self.mrstftloss = MultiResolutionSTFTLoss(
            n_bins=num_bins, sample_rate=sample_rate
        )
        self.l1loss = nn.L1Loss()

    def forward(self, batch):
        x, target = batch
        output = self.model(x.squeeze(1))  # B x T
        # Crop target to match output
        if output.shape[-1] < target.shape[-1]:
            target = causal_crop(target, output.shape[-1])
        loss = self.mrstftloss(output, target) + self.l1loss(output, target) * 100
        return loss, output

    def sample(self, x: Tensor) -> Tensor:
        output = self.model(x.squeeze(1))  # B x T
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


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """Mixup data augmentation for time-domain signals.
    Args:
        x (torch.Tensor): Batch of time-domain signals, shape [batch, 1, time].
        y (torch.Tensor): Batch of labels, shape [batch, n_classes].
        alpha (float): Beta distribution parameter.
    Returns:
        torch.Tensor: Mixed time-domain signals, shape [batch, 1, time].
        torch.Tensor: Mixed labels, shape [batch, n_classes].
        torch.Tensor: Lambda
    """
    batch_size = x.size(0)
    if alpha > 0:
        # lam = np.random.beta(alpha, alpha)
        lam = np.random.uniform(0.25, 0.75, batch_size)
        lam = torch.from_numpy(lam).float().to(x.device).view(batch_size, 1, 1)
    else:
        lam = 1

    print(lam)
    if np.random.rand() > 0.5:
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = torch.logical_or(y, y[index, :]).float()
    else:
        mixed_x = x
        mixed_y = y

    return mixed_x, mixed_y, lam


class FXClassifier(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_weight_decay: float,
        sample_rate: float,
        network: nn.Module,
        mixup: bool = False,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lr = lr
        self.lr_weight_decay = lr_weight_decay
        self.sample_rate = sample_rate
        self.network = network
        self.effects = ["Reverb", "Chorus", "Delay", "Distortion", "Compressor"]
        self.mixup = mixup
        self.label_smoothing = label_smoothing

        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.loss_fn = torch.nn.BCELoss()

        if False:
            self.train_f1 = torchmetrics.classification.MultilabelF1Score(
                5, average="none", multidim_average="global"
            )
            self.val_f1 = torchmetrics.classification.MultilabelF1Score(
                5, average="none", multidim_average="global"
            )
            self.test_f1 = torchmetrics.classification.MultilabelF1Score(
                5, average="none", multidim_average="global"
            )

            self.train_f1_avg = torchmetrics.classification.MultilabelF1Score(
                5, threshold=0.5, average="macro", multidim_average="global"
            )
            self.val_f1_avg = torchmetrics.classification.MultilabelF1Score(
                5, threshold=0.5, average="macro", multidim_average="global"
            )
            self.test_f1_avg = torchmetrics.classification.MultilabelF1Score(
                5, threshold=0.5, average="macro", multidim_average="global"
            )

            self.metrics = {
                "train": self.train_acc,
                "valid": self.val_acc,
                "test": self.test_acc,
            }

            self.avg_metrics = {
                "train": self.train_f1_avg,
                "valid": self.val_f1_avg,
                "test": self.test_f1_avg,
            }

        self.metrics = torch.nn.ModuleDict()
        for effect in self.effects:
            self.metrics[f"train_{effect}_acc"] = torchmetrics.classification.Accuracy(
                task="binary"
            )
            self.metrics[f"valid_{effect}_acc"] = torchmetrics.classification.Accuracy(
                task="binary"
            )
            self.metrics[f"test_{effect}_acc"] = torchmetrics.classification.Accuracy(
                task="binary"
            )

    def forward(self, x: torch.Tensor, train: bool = False):
        return self.network(x, train=train)

    def common_step(self, batch, batch_idx, mode: str = "train"):
        train = True if mode == "train" else False
        x, y, dry_label, wet_label = batch

        if mode == "train" and self.mixup:
            x_mixed, label_mixed, lam = mixup(x, wet_label)
            outputs = self(x_mixed, train)
            loss = 0
            for idx, output in enumerate(outputs):
                loss += self.loss_fn(output.squeeze(-1), label_mixed[..., idx])
        else:
            outputs = self(x, train)
            loss = 0
            for idx, output in enumerate(outputs):
                loss += self.loss_fn(output.squeeze(-1), wet_label[..., idx])

        self.log(
            f"{mode}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        acc_metrics = []
        for idx, effect_name in enumerate(self.effects):
            acc_metric = self.metrics[f"{mode}_{effect_name}_acc"](
                outputs[idx].squeeze(-1), wet_label[..., idx]
            )
            self.log(
                f"{mode}_{effect_name}_acc",
                acc_metric,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            acc_metrics.append(acc_metric)

        self.log(
            f"{mode}_avg_acc",
            torch.mean(torch.stack(acc_metrics)),
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
