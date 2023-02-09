import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from pathlib import Path
import pytorch_lightning as pl
from typing import Any, List, Tuple

# https://zenodo.org/record/7044411/ -> GuitarFX
# https://zenodo.org/record/3371780  -> GuitarSet


class GuitarFXDataset(Dataset):
    def __init__(
        self,
        root: str,
        sample_rate: int,
        chunk_size_in_sec: int = 3,
        effect_types: List[str] = None,
    ):
        super().__init__()
        self.wet_files = []
        self.dry_files = []
        self.chunks = []
        self.labels = []
        self.song_idx = []
        self.root = Path(root)
        self.chunk_size_in_sec = chunk_size_in_sec
        self.sample_rate = sample_rate

        if effect_types is None:
            effect_types = [
                d.name for d in self.root.iterdir() if d.is_dir() and d != "Clean"
            ]
        current_file = 0
        for i, effect in enumerate(effect_types):
            for pickup in Path(self.root / effect).iterdir():
                wet_files = sorted(list(pickup.glob("*.wav")))
                dry_files = sorted(
                    list(self.root.glob(f"Clean/{pickup.name}/**/*.wav"))
                )
                self.wet_files += wet_files
                self.dry_files += dry_files
                self.labels += [i] * len(wet_files)
                for audio_file in wet_files:
                    chunk_starts, orig_sr = create_sequential_chunks(
                        audio_file, self.chunk_size_in_sec
                    )
                    self.chunks += chunk_starts
                    self.song_idx += [current_file] * len(chunk_starts)
                    current_file += 1
        print(
            f"Found {len(self.wet_files)} wet files and {len(self.dry_files)} dry files.\n"
            f"Total chunks: {len(self.chunks)}"
        )
        self.resampler = T.Resample(orig_sr, sample_rate)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        # Load effected and "clean" audio
        song_idx = self.song_idx[idx]
        x, sr = torchaudio.load(self.wet_files[song_idx])
        y, sr = torchaudio.load(self.dry_files[song_idx])
        effect_label = self.labels[song_idx]  # Effect label

        chunk_start = self.chunks[idx]
        chunk_size_in_samples = self.chunk_size_in_sec * sr
        x = x[:, chunk_start : chunk_start + chunk_size_in_samples]
        y = y[:, chunk_start : chunk_start + chunk_size_in_samples]

        resampled_x = self.resampler(x)
        resampled_y = self.resampler(y)
        # Reset chunk size to be new sample rate
        chunk_size_in_samples = self.chunk_size_in_sec * self.sample_rate
        # Pad to chunk_size if needed
        if resampled_x.shape[-1] < chunk_size_in_samples:
            resampled_x = F.pad(
                resampled_x, (0, chunk_size_in_samples - resampled_x.shape[1])
            )
        if resampled_y.shape[-1] < chunk_size_in_samples:
            resampled_y = F.pad(
                resampled_y, (0, chunk_size_in_samples - resampled_y.shape[1])
            )
        return (resampled_x, resampled_y, effect_label)


class GuitarSet(Dataset):
    def __init__(
        self,
        root: str,
        sample_rate: int,
        chunk_size_in_sec: int = 3,
        effect_types: List[torch.nn.Module] = None,
    ):
        super().__init__()
        self.chunks = []
        self.song_idx = []
        self.root = Path(root)
        self.chunk_size_in_sec = chunk_size_in_sec
        self.files = sorted(list(self.root.glob("./**/*.wav")))
        self.sample_rate = sample_rate
        for i, audio_file in enumerate(self.files):
            chunk_starts, orig_sr = create_sequential_chunks(
                audio_file, self.chunk_size_in_sec
            )
            self.chunks += chunk_starts
            self.song_idx += [i] * len(chunk_starts)
        print(f"Found {len(self.files)} files .\n" f"Total chunks: {len(self.chunks)}")
        self.resampler = T.Resample(orig_sr, sample_rate)
        self.effect_types = effect_types

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        # Load and effect audio
        song_idx = self.song_idx[idx]
        x, sr = torchaudio.load(self.files[song_idx])
        chunk_start = self.chunks[idx]
        chunk_size_in_samples = self.chunk_size_in_sec * sr
        x = x[:, chunk_start : chunk_start + chunk_size_in_samples]
        resampled_x = self.resampler(x)
        # Reset chunk size to be new sample rate
        chunk_size_in_samples = self.chunk_size_in_sec * self.sample_rate
        # Pad to chunk_size if needed
        if resampled_x.shape[-1] < chunk_size_in_samples:
            resampled_x = F.pad(
                resampled_x, (0, chunk_size_in_samples - resampled_x.shape[1])
            )
        target = resampled_x

        # Add random effect
        random_effect_idx = torch.rand(1).item() * len(self.effect_types.keys())
        effect_name = list(self.effect_types.keys())[int(random_effect_idx)]
        effect = self.effect_types[effect_name]
        effected_input = effect(resampled_x)
        return (effected_input, target, effect_name)


def create_random_chunks(
    audio_file: str, chunk_size: int, num_chunks: int
) -> Tuple[List[Tuple[int, int]], int]:
    """Create num_chunks random chunks of size chunk_size (seconds)
    from an audio file.
    Return sample_index of start of each chunk and original sr
    """
    audio, sr = torchaudio.load(audio_file)
    chunk_size_in_samples = chunk_size * sr
    if chunk_size_in_samples >= audio.shape[-1]:
        chunk_size_in_samples = audio.shape[-1] - 1
    chunks = []
    for i in range(num_chunks):
        start = torch.randint(0, audio.shape[-1] - chunk_size_in_samples, (1,)).item()
        chunks.append(start)
    return chunks, sr


def create_sequential_chunks(
    audio_file: str, chunk_size: int
) -> Tuple[List[Tuple[int, int]], int]:
    """Create sequential chunks of size chunk_size (seconds) from an audio file.
    Return sample_index of start of each chunk and original sr
    """
    audio, sr = torchaudio.load(audio_file)
    chunk_size_in_samples = chunk_size * sr
    chunk_starts = torch.arange(0, audio.shape[-1], chunk_size_in_samples)
    return chunk_starts, sr


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
        train_size = round(split[0] * len(self.dataset))
        val_size = round(split[1] * len(self.dataset))
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
