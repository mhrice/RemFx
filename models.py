from audio_diffusion_pytorch import AudioDiffusionModel
import torch
from torch import Tensor
import pytorch_lightning as pl
from einops import rearrange
import wandb

SAMPLE_RATE = 22050  # From audio-diffusion-pytorch


class TCNWrapper(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AudioDiffusionModel(in_channels=1)

    def forward(self, x: torch.Tensor):
        return self.model(x)

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


class AudioDiffusionWrapper(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AudioDiffusionModel(in_channels=1)

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
        sampled = self.model.sample(
            noise=noise, num_steps=num_steps  # Suggested range: 2-50
        )
        self.log_wandb_audio_batch(
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
                f"sample_{idx}_{id}": wandb.Audio(
                    samples[idx].cpu().numpy(),
                    caption=caption,
                    sample_rate=sampling_rate,
                )
            }
        )
