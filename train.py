from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from datasets import GuitarFXDataset
from models import DiffusionGenerationModel, OpenUnmixModel


SAMPLE_RATE = 22050
TRAIN_SPLIT = 0.8


def main():
    wandb_logger = WandbLogger(project="RemFX", save_dir="./")
    trainer = pl.Trainer(logger=wandb_logger, max_epochs=100)
    guitfx = GuitarFXDataset(
        root="./data/egfx",
        sample_rate=SAMPLE_RATE,
        effect_type=["Phaser"],
    )
    train_size = int(TRAIN_SPLIT * len(guitfx))
    val_size = len(guitfx) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        guitfx, [train_size, val_size]
    )
    train = DataLoader(train_dataset, batch_size=2)
    val = DataLoader(val_dataset, batch_size=2)

    # model = DiffusionGenerationModel()
    model = OpenUnmixModel()

    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=val)


if __name__ == "__main__":
    main()
