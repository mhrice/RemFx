import hydra
from omegaconf import DictConfig
import torch
from remfx.models import RemFXChainInference
import torchaudio


@hydra.main(
    version_base=None,
    config_path="../cfg",
    config_name="config.yaml",
)
def main(cfg: DictConfig):
    print("Loading models...")
    models = {}
    for effect in cfg.ckpts:
        model = hydra.utils.instantiate(cfg.ckpts[effect].model, _convert_="partial")
        ckpt_path = cfg.ckpts[effect].ckpt_path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
        model.load_state_dict(state_dict)
        model.to(device)
        models[effect] = model

    classifier = hydra.utils.instantiate(cfg.classifier, _convert_="partial")
    ckpt_path = cfg.classifier_ckpt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
    classifier.load_state_dict(state_dict)
    classifier.to(device)

    inference_model = RemFXChainInference(
        models,
        sample_rate=cfg.sample_rate,
        num_bins=cfg.num_bins,
        effect_order=cfg.inference_effects_ordering,
        classifier=classifier,
        shuffle_effect_order=cfg.inference_effects_shuffle,
        use_all_effect_models=cfg.inference_use_all_effect_models,
    )

    audio_file = cfg.audio_input
    print("Loading", audio_file)
    audio, sr = torchaudio.load(audio_file)
    # Resample
    audio = torchaudio.transforms.Resample(sr, cfg.sample_rate)(audio)
    # Convert to mono
    audio = audio.mean(0, keepdim=True)
    # Add dimension for batch
    audio = audio.unsqueeze(0)
    audio = audio.to(device)
    batch = [audio, audio, None, None]

    _, y = inference_model(batch, 0, verbose=True)
    y = y.cpu()
    if "output_path" in cfg:
        output_path = cfg.output_path
    else:
        output_path = "./output.wav"
    print("Saving output to", output_path)
    torchaudio.save(output_path, y[0], sample_rate=cfg.sample_rate)


if __name__ == "__main__":
    main()
