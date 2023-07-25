import os
import sys
import glob
import torch
import shutil
import torchaudio
import pytorch_lightning as pl
import random
from tqdm import tqdm
from pathlib import Path
from remfx import effects as effect_lib
from typing import Any, List, Dict
from torch.utils.data import Dataset, DataLoader
from remfx.utils import select_random_chunk
import multiprocessing
from auraloss.freq import MultiResolutionSTFTLoss


STFT_THRESH = 1e-3
ALL_EFFECTS = effect_lib.Pedalboard_Effects


vocalset_splits = {
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

guitarset_splits = {"train": ["00", "01", "02", "03"], "val": ["04"], "test": ["05"]}
idmt_guitar_splits = {
    "train": ["classical", "country_folk", "jazz", "latin", "metal", "pop"],
    "val": ["reggae", "ska"],
    "test": ["rock", "blues"],
}
idmt_bass_splits = {
    "train": ["BE", "BEQ"],
    "val": ["VIF"],
    "test": ["VIS"],
}
dsd_100_splits = {
    "train": ["train"],
    "val": ["val"],
    "test": ["test"],
}
idmt_drums_splits = {
    "train": ["WaveDrum02", "TechnoDrum01"],
    "val": ["RealDrum01"],
    "test": ["TechnoDrum02", "WaveDrum01"],
}


def locate_files(root: str, mode: str):
    file_list = []
    # ------------------------- VocalSet -------------------------
    vocalset_dir = os.path.join(root, "VocalSet1-2")
    if os.path.isdir(vocalset_dir):
        # find all singer directories
        singer_dirs = glob.glob(os.path.join(vocalset_dir, "data_by_singer", "*"))
        singer_dirs = [
            sd for sd in singer_dirs if os.path.basename(sd) in vocalset_splits[mode]
        ]
        files = []
        for singer_dir in singer_dirs:
            files += glob.glob(os.path.join(singer_dir, "**", "**", "*.wav"))
        print(f"Found {len(files)} files in VocalSet {mode}.")
        file_list.append(sorted(files))
    # ------------------------- GuitarSet -------------------------
    guitarset_dir = os.path.join(root, "audio_mono-mic")
    if os.path.isdir(guitarset_dir):
        files = glob.glob(os.path.join(guitarset_dir, "*.wav"))
        files = [
            f
            for f in files
            if os.path.basename(f).split("_")[0] in guitarset_splits[mode]
        ]
        print(f"Found {len(files)} files in GuitarSet {mode}.")
        file_list.append(sorted(files))
    # # ------------------------- IDMT-SMT-GUITAR -------------------------
    # idmt_smt_guitar_dir = os.path.join(root, "IDMT-SMT-GUITAR_V2")
    # if os.path.isdir(idmt_smt_guitar_dir):
    #     files = glob.glob(
    #         os.path.join(
    #             idmt_smt_guitar_dir, "IDMT-SMT-GUITAR_V2", "dataset4", "**", "*.wav"
    #         ),
    #         recursive=True,
    #     )
    #     files = [
    #         f
    #         for f in files
    #         if os.path.basename(f).split("_")[0] in idmt_guitar_splits[mode]
    #     ]
    #     file_list.append(sorted(files))
    #     print(f"Found {len(files)} files in IDMT-SMT-Guitar {mode}.")
    # ------------------------- IDMT-SMT-BASS -------------------------
    # idmt_smt_bass_dir = os.path.join(root, "IDMT-SMT-BASS")
    # if os.path.isdir(idmt_smt_bass_dir):
    #     files = glob.glob(
    #         os.path.join(idmt_smt_bass_dir, "**", "*.wav"),
    #         recursive=True,
    #     )
    #     files = [
    #         f
    #         for f in files
    #         if os.path.basename(os.path.dirname(f)) in idmt_bass_splits[mode]
    #     ]
    #     file_list.append(sorted(files))
    #     print(f"Found {len(files)} files in IDMT-SMT-Bass {mode}.")
    # ------------------------- DSD100 ---------------------------------
    dsd_100_dir = os.path.join(root, "DSD100")
    if os.path.isdir(dsd_100_dir):
        files = glob.glob(
            os.path.join(dsd_100_dir, mode, "**", "*.wav"),
            recursive=True,
        )
        file_list.append(sorted(files))
        print(f"Found {len(files)} files in DSD100 {mode}.")
    # ------------------------- IDMT-SMT-DRUMS -------------------------
    idmt_smt_drums_dir = os.path.join(root, "IDMT-SMT-DRUMS-V2")
    if os.path.isdir(idmt_smt_drums_dir):
        files = glob.glob(os.path.join(idmt_smt_drums_dir, "audio", "*.wav"))
        files = [
            f
            for f in files
            if os.path.basename(f).split("_")[0] in idmt_drums_splits[mode]
        ]
        file_list.append(sorted(files))
        print(f"Found {len(files)} files in IDMT-SMT-Drums {mode}.")

    return file_list


def parallel_process_effects(
    chunk_idx: int,
    proc_root: str,
    files: list,
    chunk_size: int,
    effects: list,
    effects_to_keep: list,
    num_kept_effects: tuple,
    shuffle_kept_effects: bool,
    effects_to_remove: list,
    num_removed_effects: tuple,
    shuffle_removed_effects: bool,
    sample_rate: int,
    target_lufs_db: float,
):
    """Note: This function has an issue with random seed. It may not fully randomize the effects."""
    chunk = None
    random_dataset_choice = random.choice(files)
    while chunk is None:
        random_file_choice = random.choice(random_dataset_choice)
        chunk = select_random_chunk(random_file_choice, chunk_size, sample_rate)

    # Sum to mono
    if chunk.shape[0] > 1:
        chunk = chunk.sum(0, keepdim=True)

    dry = chunk

    # loudness normalization
    normalize = effect_lib.LoudnessNormalize(sample_rate, target_lufs_db=target_lufs_db)

    # Apply Kept Effects
    # Shuffle effects if specified
    if shuffle_kept_effects:
        effect_indices = torch.randperm(len(effects_to_keep))
    else:
        effect_indices = torch.arange(len(effects_to_keep))

    r1 = num_kept_effects[0]
    r2 = num_kept_effects[1]
    num_kept_effects = torch.round((r1 - r2) * torch.rand(1) + r2).int()
    effect_indices = effect_indices[:num_kept_effects]
    # Index in effect settings
    effect_names_to_apply = [effects_to_keep[i] for i in effect_indices]
    effects_to_apply = [effects[i] for i in effect_names_to_apply]
    # Apply
    dry_labels = []
    for effect in effects_to_apply:
        # Normalize in-between effects
        dry = normalize(effect(dry))
        dry_labels.append(ALL_EFFECTS.index(type(effect)))

    # Apply effects_to_remove
    # Shuffle effects if specified
    if shuffle_removed_effects:
        effect_indices = torch.randperm(len(effects_to_remove))
    else:
        effect_indices = torch.arange(len(effects_to_remove))
    wet = torch.clone(dry)
    r1 = num_removed_effects[0]
    r2 = num_removed_effects[1]
    num_removed_effects = torch.round((r1 - r2) * torch.rand(1) + r2).int()
    effect_indices = effect_indices[:num_removed_effects]
    # Index in effect settings
    effect_names_to_apply = [effects_to_remove[i] for i in effect_indices]
    effects_to_apply = [effects[i] for i in effect_names_to_apply]
    # Apply
    wet_labels = []
    for effect in effects_to_apply:
        # Normalize in-between effects
        wet = normalize(effect(wet))
        wet_labels.append(ALL_EFFECTS.index(type(effect)))

    wet_labels_tensor = torch.zeros(len(ALL_EFFECTS))
    dry_labels_tensor = torch.zeros(len(ALL_EFFECTS))

    for label_idx in wet_labels:
        wet_labels_tensor[label_idx] = 1.0

    for label_idx in dry_labels:
        dry_labels_tensor[label_idx] = 1.0

    # Normalize
    normalized_dry = normalize(dry)
    normalized_wet = normalize(wet)

    output_dir = proc_root / str(chunk_idx)
    output_dir.mkdir(exist_ok=True)
    torchaudio.save(output_dir / "input.wav", normalized_wet, sample_rate)
    torchaudio.save(output_dir / "target.wav", normalized_dry, sample_rate)
    torch.save(dry_labels_tensor, output_dir / "dry_effects.pt")
    torch.save(wet_labels_tensor, output_dir / "wet_effects.pt")

    # return normalized_dry, normalized_wet, dry_labels_tensor, wet_labels_tensor


class DynamicEffectDataset(Dataset):
    def __init__(
        self,
        root: str,
        sample_rate: int,
        chunk_size: int = 262144,
        total_chunks: int = 1000,
        effect_modules: List[Dict[str, torch.nn.Module]] = None,
        effects_to_keep: List[str] = None,
        effects_to_remove: List[str] = None,
        num_kept_effects: List[int] = [1, 5],
        num_removed_effects: List[int] = [1, 5],
        shuffle_kept_effects: bool = True,
        shuffle_removed_effects: bool = False,
        render_files: bool = True,
        render_root: str = None,
        mode: str = "train",
        parallel: bool = False,
    ) -> None:
        super().__init__()
        self.chunks = []
        self.song_idx = []
        self.root = Path(root)
        self.render_root = Path(render_root)
        self.chunk_size = chunk_size
        self.total_chunks = total_chunks
        self.sample_rate = sample_rate
        self.mode = mode
        self.num_kept_effects = num_kept_effects
        self.num_removed_effects = num_removed_effects
        self.effects_to_keep = [] if effects_to_keep is None else effects_to_keep
        self.effects_to_remove = [] if effects_to_remove is None else effects_to_remove
        self.normalize = effect_lib.LoudnessNormalize(sample_rate, target_lufs_db=-20)
        self.effects = effect_modules
        self.shuffle_kept_effects = shuffle_kept_effects
        self.shuffle_removed_effects = shuffle_removed_effects
        effects_string = "_".join(
            self.effects_to_keep
            + ["_"]
            + self.effects_to_remove
            + ["_"]
            + [str(x) for x in num_kept_effects]
            + ["_"]
            + [str(x) for x in num_removed_effects]
        )
        # self.validate_effect_input()
        # self.proc_root = self.render_root / "processed" / effects_string / self.mode
        self.parallel = parallel
        self.files = locate_files(self.root, self.mode)

    def process_effects(self, dry: torch.Tensor):
        # Apply Kept Effects
        # Shuffle effects if specified
        if self.shuffle_kept_effects:
            effect_indices = torch.randperm(len(self.effects_to_keep))
        else:
            effect_indices = torch.arange(len(self.effects_to_keep))

        r1 = self.num_kept_effects[0]
        r2 = self.num_kept_effects[1]
        num_kept_effects = torch.round((r1 - r2) * torch.rand(1) + r2).int()
        effect_indices = effect_indices[:num_kept_effects]
        # Index in effect settings
        effect_names_to_apply = [self.effects_to_keep[i] for i in effect_indices]
        effects_to_apply = [self.effects[i] for i in effect_names_to_apply]
        # Apply
        dry_labels = []
        for effect in effects_to_apply:
            # Normalize in-between effects
            dry = self.normalize(effect(dry))
            dry_labels.append(ALL_EFFECTS.index(type(effect)))

        # Apply effects_to_remove
        # Shuffle effects if specified
        if self.shuffle_removed_effects:
            effect_indices = torch.randperm(len(self.effects_to_remove))
        else:
            effect_indices = torch.arange(len(self.effects_to_remove))
        wet = torch.clone(dry)
        r1 = self.num_removed_effects[0]
        r2 = self.num_removed_effects[1]
        num_removed_effects = torch.round((r1 - r2) * torch.rand(1) + r2).int()
        effect_indices = effect_indices[:num_removed_effects]
        # Index in effect settings
        effect_names_to_apply = [self.effects_to_remove[i] for i in effect_indices]
        effects_to_apply = [self.effects[i] for i in effect_names_to_apply]
        # Apply
        wet_labels = []
        for effect in effects_to_apply:
            # Normalize in-between effects
            wet = self.normalize(effect(wet))
            wet_labels.append(ALL_EFFECTS.index(type(effect)))

        wet_labels_tensor = torch.zeros(len(ALL_EFFECTS))
        dry_labels_tensor = torch.zeros(len(ALL_EFFECTS))

        for label_idx in wet_labels:
            wet_labels_tensor[label_idx] = 1.0

        for label_idx in dry_labels:
            dry_labels_tensor[label_idx] = 1.0

        # Normalize
        normalized_dry = self.normalize(dry)
        normalized_wet = self.normalize(wet)
        return normalized_dry, normalized_wet, dry_labels_tensor, wet_labels_tensor

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, _: int):
        chunk = None
        random_dataset_choice = random.choice(self.files)
        while chunk is None:
            random_file_choice = random.choice(random_dataset_choice)
            chunk = select_random_chunk(
                random_file_choice, self.chunk_size, self.sample_rate
            )

        # Sum to mono
        if chunk.shape[0] > 1:
            chunk = chunk.sum(0, keepdim=True)

        dry, wet, dry_effects, wet_effects = self.process_effects(chunk)

        return wet, dry, dry_effects, wet_effects


class EffectDataset(Dataset):
    def __init__(
        self,
        root: str,
        sample_rate: int,
        chunk_size: int = 262144,
        total_chunks: int = 1000,
        effect_modules: List[Dict[str, torch.nn.Module]] = None,
        effects_to_keep: List[str] = None,
        effects_to_remove: List[str] = None,
        num_kept_effects: List[int] = [1, 5],
        num_removed_effects: List[int] = [1, 5],
        shuffle_kept_effects: bool = True,
        shuffle_removed_effects: bool = False,
        render_files: bool = True,
        render_root: str = None,
        mode: str = "train",
        parallel: bool = False,
    ):
        super().__init__()
        self.chunks = []
        self.song_idx = []
        self.root = Path(root)
        self.render_root = Path(render_root)
        self.chunk_size = chunk_size
        self.total_chunks = total_chunks
        self.sample_rate = sample_rate
        self.mode = mode
        self.num_kept_effects = num_kept_effects
        self.num_removed_effects = num_removed_effects
        self.effects_to_keep = [] if effects_to_keep is None else effects_to_keep
        self.effects_to_remove = [] if effects_to_remove is None else effects_to_remove
        self.normalize = effect_lib.LoudnessNormalize(sample_rate, target_lufs_db=-20)
        self.mrstft = MultiResolutionSTFTLoss(sample_rate=sample_rate)
        self.effects = effect_modules
        self.shuffle_kept_effects = shuffle_kept_effects
        self.shuffle_removed_effects = shuffle_removed_effects
        effects_string = "_".join(
            self.effects_to_keep
            + ["_"]
            + self.effects_to_remove
            + ["_"]
            + [str(x) for x in num_kept_effects]
            + ["_"]
            + [str(x) for x in num_removed_effects]
        )
        self.validate_effect_input()
        self.proc_root = self.render_root / "processed" / effects_string / self.mode
        self.parallel = parallel

        self.files = locate_files(self.root, self.mode)

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

        print("Total datasets:", len(self.files))
        print("Processing files...")
        if render_files:
            # Split audio file into chunks, resample, then apply random effects
            self.proc_root.mkdir(parents=True, exist_ok=True)

            if self.parallel:
                items = [
                    (
                        chunk_idx,
                        self.proc_root,
                        self.files,
                        self.chunk_size,
                        self.effects,
                        self.effects_to_keep,
                        self.num_kept_effects,
                        self.shuffle_kept_effects,
                        self.effects_to_remove,
                        self.num_removed_effects,
                        self.shuffle_removed_effects,
                        self.sample_rate,
                        -20.0,
                    )
                    for chunk_idx in range(self.total_chunks)
                ]
                with multiprocessing.Pool(processes=32) as pool:
                    pool.starmap(parallel_process_effects, items)
                print(f"Done proccessing {self.total_chunks}", flush=True)
            else:
                for num_chunk in tqdm(range(self.total_chunks)):
                    chunk = None
                    random_dataset_choice = random.choice(self.files)
                    while chunk is None:
                        random_file_choice = random.choice(random_dataset_choice)
                        chunk = select_random_chunk(
                            random_file_choice, self.chunk_size, self.sample_rate
                        )
                    # Sum to mono
                    if chunk.shape[0] > 1:
                        chunk = chunk.sum(0, keepdim=True)

                    dry, wet, dry_effects, wet_effects = self.process_effects(chunk)
                    output_dir = self.proc_root / str(num_chunk)
                    output_dir.mkdir(exist_ok=True)
                    torchaudio.save(output_dir / "input.wav", wet, self.sample_rate)
                    torchaudio.save(output_dir / "target.wav", dry, self.sample_rate)
                    torch.save(dry_effects, output_dir / "dry_effects.pt")
                    torch.save(wet_effects, output_dir / "wet_effects.pt")

            print("Finished rendering")
        else:
            self.total_chunks = len(list(self.proc_root.iterdir()))

        print("Total chunks:", self.total_chunks)

    def __len__(self):
        return self.total_chunks

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
        for effect in self.effects_to_keep:
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
        kept_str = "randomly" if self.shuffle_kept_effects else "in order"
        rem_str = "randomly" if self.shuffle_removed_effects else "in order"
        if self.num_kept_effects[0] > self.num_kept_effects[1]:
            raise ValueError(
                f"num_kept_effects must be a tuple of (min, max). "
                f"Got {self.num_kept_effects}"
            )
        if self.num_kept_effects[0] == self.num_kept_effects[1]:
            num_kept_str = f"{self.num_kept_effects[0]}"
        else:
            num_kept_str = (
                f"Between {self.num_kept_effects[0]}-{self.num_kept_effects[1]}"
            )
        if self.num_removed_effects[0] > self.num_removed_effects[1]:
            raise ValueError(
                f"num_removed_effects must be a tuple of (min, max). "
                f"Got {self.num_removed_effects}"
            )
        if self.num_removed_effects[0] == self.num_removed_effects[1]:
            num_rem_str = f"{self.num_removed_effects[0]}"
        else:
            num_rem_str = (
                f"Between {self.num_removed_effects[0]}-{self.num_removed_effects[1]}"
            )
        rem_fx = self.effects_to_remove
        kept_fx = self.effects_to_keep
        print(
            f"Effect Summary: \n"
            f"Apply kept effects: {kept_fx} ({num_kept_str}, chosen {kept_str}) -> Dry\n"
            f"Apply remove effects: {rem_fx} ({num_rem_str}, chosen {rem_str}) -> Wet\n"
        )

    def process_effects(self, dry: torch.Tensor):
        # Apply Kept Effects
        # Shuffle effects if specified
        if self.shuffle_kept_effects:
            effect_indices = torch.randperm(len(self.effects_to_keep))
        else:
            effect_indices = torch.arange(len(self.effects_to_keep))

        r1 = self.num_kept_effects[0]
        r2 = self.num_kept_effects[1]
        num_kept_effects = torch.round((r1 - r2) * torch.rand(1) + r2).int()
        effect_indices = effect_indices[:num_kept_effects]
        # Index in effect settings
        effect_names_to_apply = [self.effects_to_keep[i] for i in effect_indices]
        effects_to_apply = [self.effects[i] for i in effect_names_to_apply]
        # stft comparison
        stft = 0
        while stft < STFT_THRESH:
            # Apply
            dry_labels = []
            for effect in effects_to_apply:
                # Normalize in-between effects
                dry = self.normalize(effect(dry))
                dry_labels.append(ALL_EFFECTS.index(type(effect)))

            # Apply effects_to_remove
            # Shuffle effects if specified
            if self.shuffle_removed_effects:
                effect_indices = torch.randperm(len(self.effects_to_remove))
            else:
                effect_indices = torch.arange(len(self.effects_to_remove))
            wet = torch.clone(dry)
            r1 = self.num_removed_effects[0]
            r2 = self.num_removed_effects[1]
            num_removed_effects = torch.round((r1 - r2) * torch.rand(1) + r2).int()
            effect_indices = effect_indices[:num_removed_effects]
            # Index in effect settings
            effect_names_to_apply = [self.effects_to_remove[i] for i in effect_indices]
            effects_to_apply = [self.effects[i] for i in effect_names_to_apply]
            # Apply
            wet_labels = []
            for effect in effects_to_apply:
                # Normalize in-between effects
                wet = self.normalize(effect(wet))
                wet_labels.append(ALL_EFFECTS.index(type(effect)))

            wet_labels_tensor = torch.zeros(len(ALL_EFFECTS))
            dry_labels_tensor = torch.zeros(len(ALL_EFFECTS))

            for label_idx in wet_labels:
                wet_labels_tensor[label_idx] = 1.0

            for label_idx in dry_labels:
                dry_labels_tensor[label_idx] = 1.0

            # Normalize
            normalized_dry = self.normalize(dry)
            normalized_wet = self.normalize(wet)

            # Check STFT, pick different effects if necessary
            stft = self.mrstft(normalized_wet, normalized_dry)
        return normalized_dry, normalized_wet, dry_labels_tensor, wet_labels_tensor


class InferenceDataset(Dataset):
    def __init__(self, root: str, sample_rate: int, **kwargs):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.clean_paths = sorted(list(self.root.glob("clean/*.wav")))
        self.effected_paths = sorted(list(self.root.glob("effected/*.wav")))

    def __len__(self) -> int:
        return len(self.clean_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        clean_path = self.clean_paths[idx]
        effected_path = self.effected_paths[idx]
        clean_audio, sr = torchaudio.load(clean_path)
        clean = torchaudio.functional.resample(clean_audio, sr, self.sample_rate)
        effected_audio, sr = torchaudio.load(effected_path)
        effected = torchaudio.functional.resample(effected_audio, sr, self.sample_rate)

        # Sum to mono
        clean = torch.sum(clean, dim=0, keepdim=True)
        effected = torch.sum(effected, dim=0, keepdim=True)

        # Pad or trim effected to clean
        if effected.shape[1] > clean.shape[1]:
            effected = effected[:, : clean.shape[1]]
        elif effected.shape[1] < clean.shape[1]:
            pad_size = clean.shape[1] - effected.shape[1]
            effected = torch.nn.functional.pad(effected, (0, pad_size))

        dry_labels_tensor = torch.zeros(len(ALL_EFFECTS))
        wet_labels_tensor = torch.ones(len(ALL_EFFECTS))

        return effected, clean, dry_labels_tensor, wet_labels_tensor


class EffectDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        *,
        train_batch_size: int,
        test_batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Any = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,  # Use small, consistent batch size for testing
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
