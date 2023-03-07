import os
import sys
import glob
import torch
import shutil
import torchaudio
import pytorch_lightning as pl
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from remfx import effects
from ordered_set import OrderedSet
from typing import Any, List, Dict
from torch.utils.data import Dataset, DataLoader
from remfx.utils import create_sequential_chunks


# https://zenodo.org/record/1193957 -> VocalSet

ALL_EFFECTS = effects.Pedalboard_Effects
print(ALL_EFFECTS)


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
        chunk_size: int = 262144,
        effect_modules: List[Dict[str, torch.nn.Module]] = None,
        effects_to_use: List[str] = None,
        effects_to_remove: List[str] = None,
        max_kept_effects: int = -1,
        max_removed_effects: int = 1,
        shuffle_kept_effects: bool = True,
        shuffle_removed_effects: bool = False,
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
        self.max_kept_effects = max_kept_effects
        self.max_removed_effects = max_removed_effects
        self.effects_to_use = effects_to_use
        self.effects_to_remove = effects_to_remove
        self.normalize = effects.LoudnessNormalize(sample_rate, target_lufs_db=-20)
        self.effects = effect_modules
        self.shuffle_kept_effects = shuffle_kept_effects
        self.shuffle_removed_effects = shuffle_removed_effects
        effects_string = "_".join(self.effects_to_use + ["_"] + self.effects_to_remove)
        self.effects_to_keep = self.validate_effect_input()
        self.proc_root = self.render_root / "processed" / effects_string / self.mode

        # find all singer directories
        singer_dirs = glob.glob(os.path.join(self.root, "data_by_singer", "*"))
        singer_dirs = [
            sd for sd in singer_dirs if os.path.basename(sd) in singer_splits[mode]
        ]
        self.files = []
        for singer_dir in singer_dirs:
            self.files += glob.glob(os.path.join(singer_dir, "**", "**", "*.wav"))
        self.files = sorted(self.files)

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

                    dry, wet, dry_effects, wet_effects = self.process_effects(
                        resampled_chunk
                    )
                    output_dir = self.proc_root / str(self.num_chunks)
                    output_dir.mkdir(exist_ok=True)
                    torchaudio.save(output_dir / "input.wav", wet, self.sample_rate)
                    torchaudio.save(output_dir / "target.wav", dry, self.sample_rate)
                    torch.save(dry_effects, output_dir / "dry_effects.pt")
                    torch.save(wet_effects, output_dir / "wet_effects.pt")
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
        dry_effect_names = torch.load(self.proc_root / str(idx) / "dry_effects.pt")
        wet_effect_names = torch.load(self.proc_root / str(idx) / "wet_effects.pt")
        input, sr = torchaudio.load(input_file)
        target, sr = torchaudio.load(target_file)
        return (input, target, dry_effect_names, wet_effect_names)

    def validate_effect_input(self):
        for effect in self.effects.values():
            if type(effect) not in ALL_EFFECTS:
                raise ValueError(
                    f"Effect {effect} not found in ALL_EFFECTS. "
                    f"Please choose from {ALL_EFFECTS}"
                )
        for effect in self.effects_to_use:
            if effect not in self.effects.keys():
                raise ValueError(
                    f"Effect {effect} not found in self.effects. "
                    f"Please choose from {self.effects.keys()}"
                )
        for effect in self.effects_to_remove:
            if effect not in self.effects.keys():
                raise ValueError(
                    f"Effect {effect} not found in self.effects. "
                    f"Please choose from {self.effects.keys()}"
                )
        kept_fx = list(
            OrderedSet(self.effects_to_use) - OrderedSet(self.effects_to_remove)
        )
        kept_str = "randomly" if self.shuffle_kept_effects else "in order"
        rem_fx = self.effects_to_remove
        rem_str = "randomly" if self.shuffle_removed_effects else "in order"
        if self.max_kept_effects == -1:
            num_kept_str = len(kept_fx)
        else:
            num_kept_str = f"Up to {self.max_kept_effects}"
        if self.max_removed_effects == -1:
            num_rem_str = len(rem_fx)
        else:
            num_rem_str = f"Up to {self.max_removed_effects}"

        print(
            f"Effect Summary: \n"
            f"Apply kept effects: {kept_fx} ({num_kept_str}, chosen {kept_str}) -> Dry\n"
            f"Apply remove effects: {rem_fx} ({num_rem_str}, chosen {rem_str}) -> Wet\n"
        )
        return kept_fx

    def process_effects(self, dry: torch.Tensor):
        # Apply Kept Effects
        # Shuffle effects if specified
        if self.shuffle_kept_effects:
            effect_indices = torch.randperm(len(self.effects_to_keep))
        else:
            effect_indices = torch.arange(len(self.effects_to_keep))

        # Up to max_kept_effects
        if self.max_kept_effects != -1:
            num_kept_effects = int(torch.rand(1).item() * (self.max_kept_effects)) + 1
        else:
            num_kept_effects = len(self.effects_to_keep)
        effect_indices = effect_indices[:num_kept_effects]
        print(effect_indices)

        # Index in effect settings
        effect_names_to_apply = [self.effects_to_keep[i] for i in effect_indices]
        effects_to_apply = [self.effects[i] for i in effect_names_to_apply]
        # Apply
        dry_labels = []
        for effect in effects_to_apply:
            dry = effect(dry)
            dry_labels.append(ALL_EFFECTS.index(type(effect)))

        # Apply effects_to_remove
        # Shuffle effects if specified
        wet = torch.clone(dry)
        if self.shuffle_removed_effects:
            effect_indices = torch.randperm(len(self.effects_to_remove))
        else:
            effect_indices = torch.arange(len(self.effects_to_remove))
        # Up to max_removed_effects
        if self.max_removed_effects != -1:
            num_kept_effects = int(torch.rand(1).item() * (self.max_removed_effects))
        else:
            num_kept_effects = len(self.effects_to_remove)
        effect_indices = effect_indices[: self.max_removed_effects]
        # Index in effect settings
        effect_names_to_apply = [self.effects_to_remove[i] for i in effect_indices]
        effects_to_apply = [self.effects[i] for i in effect_names_to_apply]
        # Apply

        wet_labels = []
        for effect in effects_to_apply:
            wet = effect(wet)
            wet_labels.append(ALL_EFFECTS.index(type(effect)))

        wet_labels_tensor = torch.zeros(len(ALL_EFFECTS))
        dry_labels_tensor = torch.zeros(len(ALL_EFFECTS))

        for label_idx in wet_labels:
            wet_labels_tensor[label_idx] = 1.0

        for label_idx in dry_labels:
            dry_labels_tensor[label_idx] = 1.0

        # effects_present = torch.sum(one_hot, dim=0).float()
        print(dry_labels_tensor, wet_labels_tensor)
        # Normalize
        normalized_dry = self.normalize(dry)
        normalized_wet = self.normalize(wet)
        return normalized_dry, normalized_wet, dry_labels_tensor, wet_labels_tensor


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
