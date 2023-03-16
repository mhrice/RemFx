import os
import sys
import glob
import torch
import argparse


def download_zip_dataset(dataset_url: str, output_dir: str):
    zip_filename = os.path.basename(dataset_url)
    zip_name = zip_filename.replace(".zip", "")
    os.system(f"wget -P {output_dir} {dataset_url}")
    os.system(
        f"""unzip {os.path.join(output_dir, zip_filename)} -d {os.path.join(output_dir, zip_name)}"""
    )
    os.system(f"rm {os.path.join(output_dir, zip_filename)}")


def process_dataset(dataset_dir: str, output_dir: str):
    if dataset_dir == "VocalSet1-2":
        pass
    elif dataset_dir == "audio_mono-mic":
        pass
    elif dataset_dir == "IDMT-SMT-GUITAR_V2":
        pass
    elif dataset_dir == "IDMT-SMT-BASS":
        pass
    elif dataset_dir == "IDMT-SMT-DRUMS-V2":
        pass
    else:
        raise NotImplemented(f"Invalid dataset_dir = {dataset_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_names",
        choices=[
            "vocalset",
            "guitarset",
            "idmt-smt-guitar",
            "idmt-smt-bass",
            "idmt-smt-drums",
        ],
        nargs="+",
    )
    args = parser.parse_args()

    dataset_urls = {
        "vocalset": "https://zenodo.org/record/1442513/files/VocalSet1-2.zip",
        "guitarset": "https://zenodo.org/record/3371780/files/audio_mono-mic.zip",
        "IDMT-SMT-GUITAR_V2": "https://zenodo.org/record/7544110/files/IDMT-SMT-GUITAR_V2.zip",
        "IDMT-SMT-BASS": "https://zenodo.org/record/7188892/files/IDMT-SMT-BASS.zip",
        "IDMT-SMT-DRUMS-V2": "https://zenodo.org/record/7544164/files/IDMT-SMT-DRUMS-V2.zip",
    }

    for dataset_name, dataset_url in dataset_urls.items():
        if dataset_name in args.dataset_names:
            download_zip_dataset(dataset_url, "~/data/remfx-data")
