import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn.functional as F
from pathlib import Path
import pytorch_lightning as pl
from typing import Any, List
from remfx import effects
from tqdm import tqdm
from remfx.utils import create_sequential_chunks

# https://zenodo.org/record/1193957 -> VocalSet


class VocalSet(Dataset):
    def __init__(
        self,
        root: str,
        sample_rate: int,
        chunk_size_in_sec: int = 3,
        effect_types: List[torch.nn.Module] = None,
        render_files: bool = True,
        mode: str = "train",
    ):
        super().__init__()
        self.chunks = []
        self.song_idx = []
        self.root = Path(root)
        self.chunk_size_in_sec = chunk_size_in_sec
        self.sample_rate = sample_rate
        self.mode = mode

        mode_path = self.root / self.mode
        self.files = sorted(list(mode_path.glob("./**/*.wav")))
        self.normalize = effects.LoudnessNormalize(sample_rate, target_lufs_db=-20)
        self.effect_types = effect_types

        self.processed_root = self.root / "processed" / self.mode

        self.num_chunks = 0
        print("Total files:", len(self.files))
        print("Processing files...")
        if render_files:
            # Split audio file into chunks, resample, then apply random effects
            self.processed_root.mkdir(parents=True, exist_ok=True)
            for audio_file in tqdm(self.files, total=len(self.files)):
                chunks, orig_sr = create_sequential_chunks(
                    audio_file, self.chunk_size_in_sec
                )
                for chunk in chunks:
                    resampled_chunk = torchaudio.functional.resample(
                        chunk, orig_sr, sample_rate
                    )
                    chunk_size_in_samples = self.chunk_size_in_sec * self.sample_rate
                    if resampled_chunk.shape[-1] < chunk_size_in_samples:
                        resampled_chunk = F.pad(
                            resampled_chunk,
                            (0, chunk_size_in_samples - resampled_chunk.shape[1]),
                        )
                    # Apply effect
                    effect_idx = torch.rand(1).item() * len(self.effect_types.keys())
                    effect_name = list(self.effect_types.keys())[int(effect_idx)]
                    effect = self.effect_types[effect_name]
                    effected_input = effect(resampled_chunk)
                    # Normalize
                    normalized_input = self.normalize(effected_input)
                    normalized_target = self.normalize(resampled_chunk)

                    output_dir = self.processed_root / str(self.num_chunks)
                    output_dir.mkdir(exist_ok=True)
                    torchaudio.save(
                        output_dir / "input.wav", normalized_input, self.sample_rate
                    )
                    torchaudio.save(
                        output_dir / "target.wav", normalized_target, self.sample_rate
                    )
                    torch.save(effect_name, output_dir / "effect_name.pt")
                    self.num_chunks += 1
        else:
            self.num_chunks = len(list(self.processed_root.iterdir()))

        print(
            f"Found {len(self.files)} {self.mode} files .\n"
            f"Total chunks: {self.num_chunks}"
        )

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        input_file = self.processed_root / str(idx) / "input.wav"
        target_file = self.processed_root / str(idx) / "target.wav"
        effect_name = torch.load(self.processed_root / str(idx) / "effect_name.pt")
        input, sr = torchaudio.load(input_file)
        target, sr = torchaudio.load(target_file)
        return (input, target, effect_name)


class VocalSetDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Any = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
