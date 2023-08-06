import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../cfg", config_name="config.yaml")
def main(cfg: DictConfig):
    # Apply seed for reproducibility
    if cfg.seed:
        pl.seed_everything(cfg.seed)
    _ = hydra.utils.instantiate(cfg.datamodule, _convert_="partial")


if __name__ == "__main__":
    main()
