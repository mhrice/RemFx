import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio
from pathlib import Path
import pytorch_lightning as pl
import sys
from typing import Any, Dict
from remfx import effects
from tqdm import tqdm
from remfx.utils import create_sequential_chunks
import shutil

# https://zenodo.org/record/1193957 -> VocalSet

ALL_EFFECTS = effects.Pedalboard_Effects

singer_splits = {
    "train": [
        "male1",
        "male2",
        "male3",
        "male4",
        "male5",
        "male6",
        "male7",
        "male8",
        "male9",
        "female1",
        "female2",
        "female3",
        "female4",
        "female5",
        "female6",
        "female7",
    ],
    "val": ["male10", "female8"],
    "test": ["male11", "female9"],
}


class VocalSet(Dataset):
    def __init__(
        self,
        root: str,
        sample_rate: int,
        chunk_size: int = 3,
        applied_effects: Dict[str, torch.nn.Module] = None,
        effect_to_remove: Dict[str, torch.nn.Module] = None,
        max_effects_per_file: int = 1,
        render_files: bool = True,
        render_root: str = None,
        mode: str = "train",
    ):
        super().__init__()
        self.chunks = []
        self.song_idx = []
        self.root = Path(root)
        self.render_root = Path(render_root)
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.mode = mode
        self.max_effects_per_file = max_effects_per_file
        self.effect_to_remove = effect_to_remove
        mode_path = self.root / self.mode

        # find all singer directories
        singer_dirs = glob.glob(os.path.join(self.root, "data_by_singer", "*"))
        singer_dirs = [
            sd for sd in singer_dirs if os.path.basename(sd) in singer_splits[mode]
        ]
        self.files = []
        for singer_dir in singer_dirs:
            self.files += glob.glob(os.path.join(singer_dir, "**", "**", "*.wav"))
        self.files = sorted(self.files)

        self.normalize = effects.LoudnessNormalize(sample_rate, target_lufs_db=-20)
        self.applied_effects = applied_effects
        self.effect_to_remove_name = "_".join([e for e in self.effect_to_remove])

        effect_str = "__".join([e for e in self.applied_effects])
        effect_str += f"_{self.effect_to_remove_name}"
        self.proc_root = self.render_root / "processed" / effect_str / self.mode

        if self.proc_root.exists() and len(list(self.proc_root.iterdir())) > 0:
            print("Found processed files.")
            if render_files:
                re_render = input(
                    "WARNING: By default, will re-render files.\n"
                    "Set render_files=False to skip re-rendering.\n"
                    "Are you sure you want to re-render? (y/n): "
                )
                if re_render != "y":
                    sys.exit()
                shutil.rmtree(self.proc_root)

        self.num_chunks = 0
        print("Total files:", len(self.files))
        print("Processing files...")
        if render_files:
            # Split audio file into chunks, resample, then apply random effects
            self.proc_root.mkdir(parents=True, exist_ok=True)
            for audio_file in tqdm(self.files, total=len(self.files)):
                chunks, orig_sr = create_sequential_chunks(audio_file, self.chunk_size)
                for chunk in chunks:
                    resampled_chunk = torchaudio.functional.resample(
                        chunk, orig_sr, sample_rate
                    )
                    if resampled_chunk.shape[-1] < chunk_size:
                        # Skip if chunk is too small
                        continue

                    x, y, effect = self.process_effects(resampled_chunk)
                    output_dir = self.proc_root / str(self.num_chunks)
                    output_dir.mkdir(exist_ok=True)
                    torchaudio.save(output_dir / "input.wav", x, self.sample_rate)
                    torchaudio.save(output_dir / "target.wav", y, self.sample_rate)
                    torch.save(effect, output_dir / "effect.pt")
                    self.num_chunks += 1
        else:
            self.num_chunks = len(list(self.proc_root.iterdir()))

        print(
            f"Found {len(self.files)} {self.mode} files .\n"
            f"Total chunks: {self.num_chunks}"
        )

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        input_file = self.proc_root / str(idx) / "input.wav"
        target_file = self.proc_root / str(idx) / "target.wav"
        effect_name = torch.load(self.proc_root / str(idx) / "effect.pt")
        input, sr = torchaudio.load(input_file)
        target, sr = torchaudio.load(target_file)
        return (input, target, effect_name)

    def process_effects(self, dry: torch.Tensor):
        # Apply random number of effects up to num_effects - 1 (excluding effect_to_remove)
        if self.max_effects_per_file > 1:
            num_effects = torch.randint(self.max_effects_per_file - 1, (1,)).item()
            # Remove effect to remove from applied effects if present
            for effect in self.effect_to_remove:
                self.applied_effects.pop(effect, None)

            # Choose random effects to apply
            effect_indices = torch.randperm(len(self.applied_effects.keys()))[
                :num_effects
            ]
            effects_to_apply = [
                list(self.applied_effects.keys())[i] for i in effect_indices
            ]
            labels = []
            for effect_name in effects_to_apply:
                effect = self.applied_effects[effect_name]
                dry = effect(dry)
                labels.append(ALL_EFFECTS.index(type(effect)))

        # Apply effect_to_remove
        wet = torch.clone(dry)
        for effect_name in self.effect_to_remove:
            effect = self.effect_to_remove[effect_name]
            wet = effect(dry)
            labels.append(ALL_EFFECTS.index(type(effect)))

        # Convert labels to one-hot
        one_hot = F.one_hot(torch.tensor(labels), num_classes=len(ALL_EFFECTS))
        effects_present = torch.sum(one_hot, dim=0).float()

        # Normalize
        normalized_dry = self.normalize(dry)
        normalized_wet = self.normalize(wet)
        return normalized_dry, normalized_wet, effects_present


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
