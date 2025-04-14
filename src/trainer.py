import os
import random
import hydra
import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

SRC_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(SRC_PATH)

import sys
sys.path.append(SRC_PATH)

from model import HRVTransformer, num_params, set_seed

from data import load_dataloaders

SRC_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(SRC_PATH)


def config_preprocessing(cfg: OmegaConf) -> OmegaConf:
    """Preprocess configuration by setting default paths and sharing common parameters.

    This function ensures configuration paths are properly set and copies shared
    parameters into both data and model configurations.

    Args:
        cfg (OmegaConf): Configuration object containing data and model settings

    Returns:
        OmegaConf: Updated configuration object with default paths and shared parameters
    """
    if not os.path.exists(cfg.data.hrv_data_dir):
        print("Warning: hrv_data_dir does not exist, using default hrv_data_dir")
        cfg.data.hrv_data_dir = os.path.join(BASE_PATH, "data")
    if not os.path.exists(cfg.wandb.logs_dir):
        print("Warning: wandb.logs_dir does not exist, using default logs directory")
        cfg.wandb.logs_dir = os.path.join(BASE_PATH, "logs")

    # Merge shared config into data and model configs
    cfg.data.n_peaks_per_sample = cfg.shared.n_peaks_per_sample
    cfg.data.class_config = cfg.shared.class_config

    cfg.model.n_peaks_per_sample = cfg.shared.n_peaks_per_sample
    cfg.model.class_config = cfg.shared.class_config
    return cfg


def get_total_batches(train_loader: DataLoader, cfg: OmegaConf) -> int:
    """Calculate the total number of batches per epoch for the training dataset.

    This function determines the total number of batches either from configuration
    or by calculating based on dataset size and batch size.

    Args:
        train_loader (DataLoader): DataLoader containing the training dataset
        cfg (OmegaConf): Configuration object containing training settings

    Returns:
        int: Total number of batches per epoch
    """
    if cfg.trainer.n_batches_train:
        total_batches = cfg.trainer.n_batches_train
    else:
        total_batches = len(train_loader.dataset) // cfg.data.train.batch_size
        if len(train_loader.dataset) % cfg.data.train.batch_size != 0:
            total_batches += 1
    return total_batches


def save_checkpoint(
    model: torch.nn.Module,
    cfg: OmegaConf,
    filename: str,
    val_loss: Optional[float] = None,
    save_config: bool = False,
    log_message: Optional[str] = None,
) -> None:
    """Save model checkpoint and optionally configuration to disk.

    This function handles saving model checkpoints to a specified directory structure,
    optionally saving configuration files, and logging to wandb.

    Args:
        model (torch.nn.Module): The model to save
        cfg (OmegaConf): Configuration object containing paths and settings
        filename (str): Name of the checkpoint file to save
        val_loss (Optional[float], optional): Validation loss for logging. Defaults to None.
        save_config (bool, optional): Whether to save config alongside model. Defaults to False.
        log_message (Optional[str], optional): Custom message to log after saving. Defaults to None.
    """
    # Get run name from wandb
    run_name = wandb.run.name if wandb.run.name else "unnamed_run"

    # Create checkpoint directory
    checkpoint_dir = os.path.join(cfg.trainer.logs_dir, "saved_models", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(model.state_dict(), checkpoint_path)

    # Save config if requested
    if save_config:
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)
        wandb.save(config_path)

    # Log to wandb
    wandb.save(checkpoint_path)

    # Log message
    if log_message:
        print(log_message)
    elif val_loss is not None:
        print(f"Saved new best model with validation loss: {val_loss:.4f}")
    else:
        print(f"Saved model checkpoint: {filename}")

    print(f"Checkpoint saved to: {checkpoint_dir}")


def run_validation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    cfg: OmegaConf,
    best_val_loss: float,
) -> Tuple[Dict[str, float], float]:
    """Run validation on the provided model and dataset.

    This function performs validation by running the model on validation data,
    collecting metrics, calculating confusion matrix values, and saving the best model
    if validation loss improves.

    Args:
        model (torch.nn.Module): Model to validate
        val_loader (DataLoader): DataLoader containing validation data
        device (torch.device): Device to run validation on
        cfg (OmegaConf): Configuration object
        best_val_loss (float): Best validation loss achieved so far

    Returns:
        Tuple[Dict[str, float], float]: A tuple containing:
            - Dictionary of validation metrics (loss, f1, f2, precision, recall)
            - Updated best validation loss
    """
    model.eval()
    total_loss = 0
    all_batch_metrics = []
    steps = 0

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0], batch[1]  # ignore ID if present
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = model.calculate_loss(outputs, y)

            # Calculate metrics
            batch_metrics = model.calculate_metrics(outputs, y)
            batch_metrics["loss"] = loss.item()  # Add loss to metrics
            all_batch_metrics.append(batch_metrics)
            total_loss += batch_metrics["loss"]
            steps += 1

    # Average metrics from all batches
    val_metrics = {}
    for key in all_batch_metrics[0].keys():
        val_metrics[key] = sum(batch[key] for batch in all_batch_metrics) / steps

    # Save best model based on validation loss
    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        save_checkpoint(
            model=model,
            cfg=cfg,
            filename="best_model.pth",
            val_loss=best_val_loss,
            save_config=True,
        )

    return val_metrics, best_val_loss


def train_epoch(
    model: torch.nn.Module,
    train_batches: List,
    val_batches: List,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Dict[str, torch.optim.lr_scheduler._LRScheduler]],
    device: torch.device,
    epoch: int,
    cfg: OmegaConf,
    global_step: int,
    best_val_loss: float,
) -> Tuple[int, float]:
    """Train the model for one complete epoch.

    This function trains the model for one epoch using the provided batches,
    periodically logs metrics, runs validation, and saves checkpoints.

    Args:
        model (torch.nn.Module): The model to train
        train_batches (List): List of training data batches
        val_batches (List): List of validation data batches
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters
        scheduler (Optional[Dict[str, torch.optim.lr_scheduler._LRScheduler]]): Optional learning rate scheduler
        device (torch.device): Device to run training on
        epoch (int): Current epoch number
        cfg (OmegaConf): Configuration object
        global_step (int): Current global step count
        best_val_loss (float): Best validation loss achieved so far

    Returns:
        Tuple[int, float]: A tuple containing:
            - Updated global step count
            - Updated best validation loss
    """
    steps = 0
    epoch_train_loss = 0.0  # Track average training loss for the epoch

    print(f"Training epoch {epoch+1}")
    for batch in train_batches:
        x, y = batch[0].to(device), batch[1].to(device)

        outputs = model(x)
        loss = model.calculate_loss(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            # Step the warmup scheduler
            if global_step < 1000:
                scheduler["warmup"].step()
            # The plateau scheduler will be stepped at the end of epoch based on train loss

        # Calculate metrics for step-level logging only
        batch_metrics = model.calculate_metrics(outputs, y)
        batch_metrics["loss"] = loss.item()  # Add loss to metrics
        epoch_train_loss += loss.item()  # Accumulate training loss

        steps += 1
        global_step += 1

        # Log every n steps
        if steps % cfg.trainer.log_every_n_steps == 0:
            wandb.log(
                {
                    "train/step": global_step,
                    "train/step_loss": batch_metrics["loss"],
                    "train/step_f1": batch_metrics["f1"],
                    "train/step_f2": batch_metrics["f2"],
                    "train/step_precision": batch_metrics["precision"],
                    "train/step_recall": batch_metrics["recall"],
                    "train/learning_rate": (
                        cfg.trainer.optim.lr
                        if not scheduler
                        else scheduler["warmup"].get_last_lr()[0]
                    ),
                    "epoch": epoch,
                },
                step=global_step,
            )

        # Run validation if needed
        if (
            cfg.trainer.use_validation
            and global_step % cfg.trainer.run_validation_every_n_steps == 0
        ):
            print(f"Running Validation, epoch {epoch}, global_step: {global_step}")
            model.eval()
            val_metrics, best_val_loss = run_validation(
                model=model,
                val_loader=val_batches,
                device=device,
                cfg=cfg,
                best_val_loss=best_val_loss,
            )

            # Log validation metrics
            wandb.log(
                {
                    "val/step_loss": val_metrics["loss"],
                    "val/step_f1": val_metrics["f1"],
                    "val/step_f2": val_metrics["f2"],
                    "val/step_precision": val_metrics["precision"],
                    "val/step_recall": val_metrics["recall"],
                },
                step=global_step,
            )

            model.train()

    # Calculate average training loss for the epoch
    avg_train_loss = epoch_train_loss / steps

    # Step the plateau scheduler based on training loss at the end of epoch
    if scheduler is not None and global_step >= 1000:
        scheduler["plateau"].step(avg_train_loss)

    # Save model checkpoint after each epoch
    save_checkpoint(
        model=model,
        cfg=cfg,
        filename=f"model_epoch_{epoch+1}.pth",
        log_message=f"Saved model checkpoint for epoch {epoch+1}",
    )

    print(f"\nCompleted epoch {epoch+1}")

    return global_step, best_val_loss


@hydra.main(config_path=SRC_PATH, config_name="config", version_base=None)
def main(cfg: OmegaConf) -> None:
    """Main training function that orchestrates the entire training process.

    This hydra-based main function handles initialization, training loop,
    evaluation, and cleanup for the model training process.

    Args:
        cfg (OmegaConf): Hydra configuration object containing all settings
    """
    set_seed()

    cfg = config_preprocessing(cfg)

    # Set device
    device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = HRVTransformer(cfg.model).to(device)
    cfg.model.num_params = num_params(model)

    # Initialize wandb
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Initialize data loaders
    train_batches, val_batches = load_dataloaders(
        cfg.data,
        n_batches_train=cfg.trainer.n_batches_train,
        n_batches_val=cfg.trainer.n_batches_val,
    )

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.trainer.optim.lr,
        weight_decay=cfg.trainer.optim.weight_decay,
    )

    scheduler = None
    if cfg.trainer.optim.use_lr_scheduler:
        # First create the warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=1000,  # 1000 steps warmup
            end_factor=1.0,
        )

        # Then create the plateau scheduler
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # Combine both schedulers
        scheduler = {"warmup": warmup_scheduler, "plateau": plateau_scheduler}

    # Training loop
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(cfg.trainer.epochs):
        # random shuffle batches if needed
        if cfg.data.train.shuffle and isinstance(train_batches, list):
            random.shuffle(train_batches)
        if cfg.data.val.shuffle and isinstance(val_batches, list):
            random.shuffle(val_batches)

        # Training phase
        model.train()
        global_step, best_val_loss = train_epoch(
            model,
            train_batches,
            val_batches,
            optimizer,
            scheduler,
            device,
            epoch,
            cfg,
            global_step,
            best_val_loss,
        )

        print(f"\nCompleted epoch {epoch+1}")

    wandb.finish()


if __name__ == "__main__":
    main()
