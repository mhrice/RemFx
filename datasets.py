import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from pathlib import Path
from typing import List

# https://zenodo.org/record/7044411/

LENGTH = 2**18  # 12 seconds
ORIG_SR = 48000


class GuitarFXDataset(Dataset):
    def __init__(
        self,
        root: str,
        sample_rate: int,
        length: int = LENGTH,
        effect_type: List[str] = None,
    ):
        self.length = length
        self.wet_files = []
        self.dry_files = []
        self.labels = []
        self.root = Path(root)
        if effect_type is None:
            effect_type = [
                d.name for d in self.root.iterdir() if d.is_dir() and d != "Clean"
            ]
        for i, effect in enumerate(effect_type):
            for pickup in Path(self.root / effect).iterdir():
                self.wet_files += list(pickup.glob("*.wav"))
                self.dry_files += list(self.root.glob(f"Clean/{pickup.name}/**/*.wav"))
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
