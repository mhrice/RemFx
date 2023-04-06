from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from einops import rearrange
import torch
import wandb
from torch import Tensor


class AudioCallback(Callback):
    def __init__(self, sample_rate, log_audio, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_audio = log_audio
        self.log_train_audio = True
        self.sample_rate = sample_rate
        if not self.log_audio:
            self.log_train_audio = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Log initial audio
        if self.log_train_audio:
            x, y, _, _ = batch
            # Concat samples together for easier viewing in dashboard
            input_samples = rearrange(x, "b c t -> c (b t)").unsqueeze(0)
            target_samples = rearrange(y, "b c t -> c (b t)").unsqueeze(0)

            log_wandb_audio_batch(
                logger=trainer.logger,
                id="input_effected_audio",
                samples=input_samples.cpu(),
                sampling_rate=self.sample_rate,
                caption="Training Data",
            )
            log_wandb_audio_batch(
                logger=trainer.logger,
                id="target_audio",
                samples=target_samples.cpu(),
                sampling_rate=self.sample_rate,
                caption="Target Data",
            )
            self.log_train_audio = False

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        x, target, _, _ = batch
        # Only run on first batch
        if batch_idx == 0 and self.log_audio:
            with torch.no_grad():
                # Avoids circular import
                from remfx.models import RemFXChainInference

                if type(pl_module) == RemFXChainInference:
                    y = pl_module.sample(batch)
                else:
                    y = pl_module.model.sample(x)
            # Concat samples together for easier viewing in dashboard
            # 2 seconds of silence between each sample
            silence = torch.zeros_like(x)
            silence = silence[:, : self.sample_rate * 2]

            concat_samples = torch.cat([y, silence, x, silence, target], dim=-1)
            log_wandb_audio_batch(
                logger=trainer.logger,
                id="prediction_input_target",
                samples=concat_samples.cpu(),
                sampling_rate=self.sample_rate,
                caption=f"Epoch {trainer.current_epoch}",
            )

    def on_test_batch_start(self, *args):
        self.on_validation_batch_start(*args)


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
