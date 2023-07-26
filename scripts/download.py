import os
import argparse
import shutil


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
    elif dataset_dir == "IDMT-SMT-BASS":
        pass
    elif dataset_dir == "IDMT-SMT-DRUMS-V2":
        pass
    elif dataset_dir == "DSD100":
        shutil.rmtree(os.path.join(output_dir, dataset_dir, "Mixtures"))
        for dir in os.listdir(os.path.join(output_dir, dataset_dir, "Sources", "Dev")):
            source = os.path.join(output_dir, dataset_dir, "Sources", "Dev", dir)
            shutil.move(source, os.path.join(output_dir, dataset_dir))
        shutil.rmtree(os.path.join(output_dir, dataset_dir, "Sources", "Dev"))
        for dir in os.listdir(os.path.join(output_dir, dataset_dir, "Sources", "Test")):
            source = os.path.join(output_dir, dataset_dir, "Sources", "Test", dir)
            shutil.move(source, os.path.join(output_dir, dataset_dir))
        shutil.rmtree(os.path.join(output_dir, dataset_dir, "Sources", "Test"))
        shutil.rmtree(os.path.join(output_dir, dataset_dir, "Sources"))

        os.mkdir(os.path.join(output_dir, dataset_dir, "train"))
        os.mkdir(os.path.join(output_dir, dataset_dir, "val"))
        os.mkdir(os.path.join(output_dir, dataset_dir, "test"))
        files = os.listdir(os.path.join(output_dir, dataset_dir))

        num = 0
        for dir in files:
            if not os.path.isdir(os.path.join(output_dir, dataset_dir, dir)):
                continue
            if dir == "train" or dir == "val" or dir == "test":
                continue
            source = os.path.join(output_dir, dataset_dir, dir, "bass.wav")
            if num < 80:
                dest = os.path.join(output_dir, dataset_dir, "train", f"{num}.wav")
            elif num < 90:
                dest = os.path.join(output_dir, dataset_dir, "val", f"{num}.wav")
            else:
                dest = os.path.join(output_dir, dataset_dir, "test", f"{num}.wav")
            shutil.move(source, dest)
            shutil.rmtree(os.path.join(output_dir, dataset_dir, dir))
            num += 1

    else:
        raise NotImplementedError(f"Invalid dataset_dir = {dataset_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_names",
        choices=[
            "vocalset",
            "guitarset",
            "dsd100",
            "idmt-smt-drums",
        ],
        nargs="+",
    )
    parser.add_argument("--output_dir", default="./data/remfx-data")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset_urls = {
        "vocalset": "https://zenodo.org/record/1442513/files/VocalSet1-2.zip",
        "guitarset": "https://zenodo.org/record/3371780/files/audio_mono-mic.zip",
        "DSD100": "http://liutkus.net/DSD100.zip",
        "IDMT-SMT-DRUMS-V2": "https://zenodo.org/record/7544164/files/IDMT-SMT-DRUMS-V2.zip",
    }

    for dataset_name, dataset_url in dataset_urls.items():
        if dataset_name in args.dataset_names:
            download_zip_dataset(dataset_url, args.output_dir)
            process_dataset(dataset_name, args.ou)
