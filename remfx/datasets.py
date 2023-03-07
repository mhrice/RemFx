import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio
from pathlib import Path
import pytorch_lightning as pl
import sys
from typing import Any, List, Dict
from remfx import effects
from tqdm import tqdm
from remfx.utils import create_sequential_chunks
import shutil


# https://zenodo.org/record/1193957 -> VocalSet

ALL_EFFECTS = effects.Pedalboard_Effects


class VocalSet(Dataset):
    def __init__(
        self,
        root: str,
        sample_rate: int,
        chunk_size: int = 3,
        effect_modules: List[Dict[str, torch.nn.Module]] = None,
        effects_to_use: List[str] = None,
        effects_to_remove: List[str] = None,
        max_kept_effects: int = 1,
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
        mode_path = self.root / self.mode
        self.files = sorted(list(mode_path.glob("./**/*.wav")))
        self.max_kept_effects = max_kept_effects
        self.effects_to_use = effects_to_use
        self.effects_to_remove = effects_to_remove
        self.normalize = effects.LoudnessNormalize(sample_rate, target_lufs_db=-20)
        self.effects = effect_modules
        self.shuffle_kept_effects = shuffle_kept_effects
        self.shuffle_removed_effects = shuffle_removed_effects

        effects_string = "_".join(self.effects_to_use + ["_"] + self.effects_to_remove)
        self.effects_to_keep = self.validate_effect_input()
        self.proc_root = self.render_root / "processed" / effects_string / self.mode

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
        kept_fx = list(set(self.effects_to_use) - set(self.effects_to_remove))
        kept_str = "randomly" if self.shuffle_kept_effects else "in order"
        removed_str = "randomly" if self.shuffle_removed_effects else "in order"
        rem_fx = self.effects_to_remove
        print(
            f"Effect Summary: \n"
            f"Apply effects: {kept_fx} (Up to {self.max_kept_effects}, chosen {kept_str}) -> Dry\n"
            f"Apply effects: {rem_fx} (All {len(rem_fx)}, chosen {removed_str}) -> Wet\n"
        )
        return kept_fx

    def process_effects(self, dry: torch.Tensor):
        labels = []

        # Apply Kept Effects
        if self.shuffle_kept_effects:
            effect_indices = torch.randperm(len(self.effects_to_keep))
        else:
            effect_indices = torch.arange(len(self.effects_to_keep))
        effect_names_to_apply = [self.effects_to_keep[i] for i in effect_indices]
        effects_to_apply = [self.effects[i] for i in effect_names_to_apply]
        for effect in effects_to_apply:
            dry = effect(dry)
            labels.append(ALL_EFFECTS.index(type(effect)))
        print(labels)

        # Apply effects_to_remove
        wet = torch.clone(dry)
        if self.shuffle_removed_effects:
            effect_indices = torch.randperm(len(self.effects_to_remove))
        else:
            effect_indices = torch.arange(len(self.effects_to_remove))
        effect_names_to_apply = [self.effects_to_remove[i] for i in effect_indices]
        effects_to_apply = [self.effects[i] for i in effect_names_to_apply]
        for effect in effects_to_apply:
            wet = effect(wet)
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
