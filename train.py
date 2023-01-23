from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import GuitarFXDataset
from remfx.models import DiffusionGenerationModel, OpenUnmixModel
import hydra
from omegaconf import DictConfig
import remfx.utils as utils

log = utils.get_logger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):
    # Apply seed for reproducibility
    print(cfg)
    pl.seed_everything(cfg.seed)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>.")
    datamodule = hydra.utils.instantiate(cfg.datamodule, _convert_="partial")

    log.info(f"Instantiating model <{cfg.model._target_}>.")
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")

    # Init all callbacks
    callbacks = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>.")
                callbacks.append(hydra.utils.instantiate(cb_conf, _convert_="partial"))

    logger = hydra.utils.instantiate(cfg.logger, _convert_="partial")
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>.")
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=cfg,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
