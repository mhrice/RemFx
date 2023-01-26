from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from pathlib import Path
import pytorch_lightning as pl
from typing import Any, List

# https://zenodo.org/record/7044411/

LENGTH = 2**18  # 12 seconds
ORIG_SR = 48000


class GuitarFXDataset(Dataset):
    def __init__(
        self,
        root: str,
        sample_rate: int,
        length: int = LENGTH,
        effect_types: List[str] = None,
    ):
        self.length = length
        self.wet_files = []
        self.dry_files = []
        self.labels = []
        self.root = Path(root)

        if effect_types is None:
            effect_types = [
                d.name for d in self.root.iterdir() if d.is_dir() and d != "Clean"
            ]
        for i, effect in enumerate(effect_types):
            for pickup in Path(self.root / effect).iterdir():
                self.wet_files += sorted(list(pickup.glob("*.wav")))
                self.dry_files += sorted(
                    list(self.root.glob(f"Clean/{pickup.name}/**/*.wav"))
                )
                self.labels += [i] * len(self.wet_files)
        print(
            f"Found {len(self.wet_files)} wet files and {len(self.dry_files)} dry files"
        )
        self.resampler = T.Resample(ORIG_SR, sample_rate)

    def __len__(self):
        return len(self.dry_files)

    def __getitem__(self, idx):
        x, sr = torchaudio.load(self.wet_files[idx])
        y, sr = torchaudio.load(self.dry_files[idx])
        effect_label = self.labels[idx]

        resampled_x = self.resampler(x)
        resampled_y = self.resampler(y)
        # Pad or crop to length
        if resampled_x.shape[-1] < self.length:
            resampled_x = F.pad(resampled_x, (0, self.length - resampled_x.shape[1]))
        elif resampled_x.shape[-1] > self.length:
            resampled_x = resampled_x[:, : self.length]
        if resampled_y.shape[-1] < self.length:
            resampled_y = F.pad(resampled_y, (0, self.length - resampled_y.shape[1]))
        elif resampled_y.shape[-1] > self.length:
            resampled_y = resampled_y[:, : self.length]
        return (resampled_x, resampled_y, effect_label)


class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        *,
        val_split: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_train: Any = None
        self.data_val: Any = None

    def setup(self, stage: Any = None) -> None:
        split = [1.0 - self.val_split, self.val_split]
        train_size = int(split[0] * len(self.dataset))
        val_size = int(split[1] * len(self.dataset))
        self.data_train, self.data_val = random_split(
            self.dataset, [train_size, val_size]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
