import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
import remfx.utils as utils
import torch
from remfx.models import RemFXChainInference

log = utils.get_logger(__name__)


@hydra.main(version_base=None, config_path="../cfg", config_name="config.yaml")
def main(cfg: DictConfig):
    # Apply seed for reproducibility
    if cfg.seed:
        pl.seed_everything(cfg.seed)
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>.")
    datamodule = hydra.utils.instantiate(cfg.datamodule, _convert_="partial")
    log.info(f"Instantiating model <{cfg.model._target_}>.")
    models = {}
    for effect in cfg.ckpts:
        ckpt_path = cfg.ckpts[effect]
        model = hydra.utils.instantiate(cfg.model, _convert_="partial")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
        model.load_state_dict(state_dict)
        model.to(device)
        models[effect] = model

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

    log.info("Instantiating Inference Model")
    inference_model = RemFXChainInference(
        models,
        sample_rate=cfg.sample_rate,
        num_bins=cfg.num_bins,
        effect_order=cfg.inference_effects_ordering,
    )
    trainer.test(model=inference_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
