import os
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    StochasticWeightAveraging,
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from typing import Union, Tuple, List, Optional, Any
from pathlib import Path
import argparse
from os.path import join, dirname

import sys
from pathlib import Path

# Add src directory to path
src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from model import HRVTransformer
from data import HRVDataModule


def build_callbacks(callback_args: OmegaConf) -> List[L.Callback]:
    """Build PyTorch Lightning callbacks based on configuration.

    Creates a list of callback objects for training, including early stopping,
    stochastic weight averaging, and learning rate monitoring, based on the
    configuration settings.

    Args:
        callback_args (OmegaConf): Configuration containing callback settings

    Returns:
        List[L.Callback]: List of Lightning callback objects
    """
    callbacks = []
    # Early Stopping
    if callback_args.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=callback_args.early_stopping.monitor,
                patience=callback_args.early_stopping.patience,
                min_delta=callback_args.early_stopping.min_delta,
                mode=callback_args.early_stopping.mode,
                verbose=callback_args.early_stopping.verbose,
            )
        )

    # Stochastic Weight Averaging
    if callback_args.stochastic_weight_averaging.enabled:
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=callback_args.stochastic_weight_averaging.lrs,
                swa_epoch_start=callback_args.stochastic_weight_averaging.swa_epoch_start,
                annealing_epochs=callback_args.stochastic_weight_averaging.annealing_epochs,
                annealing_strategy=callback_args.stochastic_weight_averaging.annealing_strategy,
            )
        )

    # Learning Rate Monitor
    if callback_args.learning_rate_monitor.enabled:
        callbacks.append(
            LearningRateMonitor(
                logging_interval=callback_args.learning_rate_monitor.logging_interval,
                log_momentum=callback_args.learning_rate_monitor.log_momentum,
            )
        )

    return callbacks


def build_logger(logger_args: OmegaConf) -> WandbLogger:
    """Build Weights & Biases (wandb) logger for experiment tracking.

    Creates a WandbLogger instance based on configuration settings, handling both
    new experiments and resuming previous runs.

    Args:
        logger_args (OmegaConf): Configuration containing logger settings including
            project name, team name, run name, and checkpoint ID for resuming

    Returns:
        WandbLogger: Configured wandb logger for experiment tracking
    """
    if logger_args.logs_dir == "logs" or logger_args.logs_dir is None:
        logger_args.logs_dir = os.path.join(os.path.dirname(src_path), "logs/")
    if logger_args.checkpoint_id:
        print(f"Loading Wandb Run...: {logger_args.checkpoint_id}")

        wandb_logger = WandbLogger(
            project=logger_args.project_name,
            entity=logger_args.team_name,
            save_dir=logger_args.logs_dir,
            resume="must",
            id=logger_args.checkpoint_id,
        )
    else:
        wandb_logger = WandbLogger(
            name=logger_args.run_name,
            project=logger_args.project_name,
            entity=logger_args.team_name,
            save_dir=logger_args.logs_dir,
        )
    return wandb_logger


def load_config(config_path: str) -> OmegaConf:
    """Load configuration from a YAML file using OmegaConf.

    Reads and parses a YAML configuration file into an OmegaConf object,
    with error handling for missing files and invalid formats.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        OmegaConf: Parsed configuration object

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the config file is invalid or cannot be parsed
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    try:
        return OmegaConf.load(config_path)
    except Exception as e:
        raise ValueError(f"Failed to load config file: {str(e)}")


def main(cfg: OmegaConf, mode: str = "train") -> Optional[Tuple[Any, ...]]:
    """Main function to run training or return experiment components.

    Handles model creation, data module setup, logger initialization, and trainer
    configuration. Can run in three modes:
    - train: Run full training
    - exp_no_trainer: Return model and data module only
    - exp_with_trainer: Return model, data module, trainer, and logger

    Args:
        cfg (OmegaConf): Configuration object containing all settings
        mode (str, optional): Operation mode. Defaults to "train".
            Options: "train", "exp_no_trainer", "exp_with_trainer"

    Returns:
        Optional[Tuple[Any, ...]]: Based on mode:
            - train: None (after training completes)
            - exp_no_trainer: (model, data_module)
            - exp_with_trainer: (model, data_module, trainer, wandb_logger)

    Raises:
        AssertionError: If mode is not one of the valid options
    """
    assert mode in ["train", "exp_no_trainer", "exp_with_trainer"]
    if cfg.logger.checkpoint_id:
        ckpt_dir = join(
            cfg.logger.logs_dir,
            cfg.logger.project_name,
            cfg.logger.checkpoint_id,
            "checkpoints",
        )
        ckpt_path = join(ckpt_dir, listdir(ckpt_dir)[-1])
    else:
        ckpt_path = None

    # ___________ Data ___________________ #
    print("Loading DataModule...")
    hrv_data_module = HRVDataModule(cfg.data)

    # ___________ Model ___________________ #
    print("Initializing Model...")
    model = HRVTransformer(cfg.model)

    if mode == "exp_no_trainer":
        if cfg.logger.checkpoint_id:
            print(f"Loading Checkpoint from id: {cfg.logger.checkpoint_id}...")
            model = model.load_from_checkpoint(cfg.logger.checkpoint_id)
        return (model, hrv_data_module)

    # ___________ Wandb Logger ___________________ #
    print("Initializing Wandb Logger...")
    wandb_logger = build_logger(cfg.logger)

    print("Initializing Trainer...")
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        strategy=cfg.trainer.strategy,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        detect_anomaly=cfg.trainer.detect_anomaly,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        logger=wandb_logger,
        max_epochs=cfg.trainer.epochs,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        deterministic=cfg.trainer.deterministic,
        callbacks=build_callbacks(cfg.trainer.callbacks),
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        overfit_batches=cfg.trainer.overfit_batches,
    )

    # Wandb object only available to rank 0
    if trainer.global_rank == 0:
        # cfg.model.num_params = model.num_params()
        # Add all config params
        wandb_logger.experiment.config.update(
            OmegaConf.to_container(cfg, resolve=True), allow_val_change=True
        )

    if mode == "exp_with_trainer":
        return (model, hrv_data_module, trainer, wandb_logger)

    print("Starting Training...")

    try:
        trainer.fit(
            model=model,
            datamodule=hrv_data_module,
            ckpt_path=ckpt_path,
        )
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        wandb_logger.experiment.log({"error": "Keyboard Interrupt"})
        return None
    except Exception as e:
        print(f"Error: {e}")
        wandb_logger.experiment.log({"error": str(e)})
        return None
    return None


if __name__ == "__main__":
    """Command-line entry point for the training script.

    Parses command-line arguments, loads the configuration file,
    and starts the training process.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=join(dirname(__file__), "config/config.yaml"),
        help="Path for the config YAML file",
    )

    args = arg_parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg=cfg, mode="train")
