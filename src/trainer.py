import os
import hydra
import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from model import HRVTransformer, num_params, set_seed
from data import load_dataloaders

SRC_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(SRC_PATH)


def config_preprocessing(cfg: OmegaConf) -> OmegaConf:
    """Check and set default values for config paths.

    Args:
        cfg: Configuration object containing data and wandb settings

    Returns:
        Updated configuration object
    """
    if cfg.data.hrv_data_dir == "":
        print("Warning: hrv_data_dir is not set, using default hrv_data_dir")
        cfg.data.hrv_data_dir = os.path.join(BASE_PATH, "data")
    if cfg.wandb.logs_dir == "":
        print("Warning: wandb.logs_dir is not set, using default logs directory")
        cfg.wandb.logs_dir = os.path.join(BASE_PATH, "logs")

    # Merge shared config into data and model configs
    cfg.data.n_peaks_per_sample = cfg.shared.n_peaks_per_sample
    cfg.data.class_config = cfg.shared.class_config

    cfg.model.n_peaks_per_sample = cfg.shared.n_peaks_per_sample
    cfg.model.class_config = cfg.shared.class_config
    return cfg


def get_total_batches(train_loader: DataLoader, cfg: OmegaConf) -> int:
    """Calculate total number of batches per epoch.

    Args:
        train_loader: DataLoader for training data
        cfg: Configuration object containing training settings

    Returns:
        Total number of batches
    """
    if cfg.trainer.n_batches_train:
        total_batches = cfg.trainer.n_batches_train
    else:
        total_batches = len(train_loader.dataset) // cfg.data.train.batch_size
        if len(train_loader.dataset) % cfg.data.train.batch_size != 0:
            total_batches += 1
    return total_batches


def get_checkpoint_dir(cfg: OmegaConf, run_name: str) -> str:
    """Get directory path for saving model checkpoints.

    Args:
        cfg: Configuration object
        run_name: Name of the current wandb run

    Returns:
        Path to checkpoint directory
    """
    return os.path.join(cfg.trainer.logs_dir, "saved_models", run_name)


def save_checkpoint(model: torch.nn.Module, cfg: OmegaConf, val_loss: float) -> None:
    """Save model checkpoint in organized directory structure.

    Args:
        model: Model to save
        cfg: Configuration object
        val_loss: Validation loss for logging
    """
    # Get run name from wandb
    run_name = wandb.run.name if wandb.run.name else "unnamed_run"

    # Create checkpoint directory
    checkpoint_dir = get_checkpoint_dir(cfg, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    torch.save(model.state_dict(), checkpoint_path)

    # Save config
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)

    # Log to wandb
    wandb.save(checkpoint_path)
    wandb.save(config_path)

    print(f"Saved new best model with validation loss: {val_loss:.4f}")
    print(f"Checkpoint saved to: {checkpoint_dir}")


def run_validation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    cfg: OmegaConf,
    best_val_loss: float,
) -> Tuple[Dict[str, float], float]:
    """Run validation if needed and return metrics and updated best loss.

    Args:
        model: Model to validate
        val_loader: DataLoader or List of fixed validation batches
        device: Device to run validation on
        cfg: Configuration object
        best_val_loss: Best validation loss so far

    Returns:
        Tuple of (validation metrics dict, updated best validation loss)
    """

    model.eval()
    total_loss = 0
    steps = 0

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0], batch[1]  # ignore ID if present
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = model.calculate_loss(outputs, y)

            # Calculate metrics
            batch_metrics = model.calculate_metrics(outputs, y, loss)
            total_loss += batch_metrics["loss"]
            steps += 1

    # Get final validation metrics
    val_metrics = {
        "loss": total_loss / steps,
        "f1": model.f1_metric.compute(),
        "precision": model.precision_metric.compute(),
        "recall": model.recall_metric.compute(),
    }

    # Reset metrics
    model.reset_metrics()

    # Save best model based on validation loss
    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        save_checkpoint(model, cfg, best_val_loss)

    return val_metrics, best_val_loss


def get_fixed_batches(loader: DataLoader, n_batches: Optional[int] = None) -> List:
    """Get fixed batches from loader to use across epochs.

    Args:
        loader: Data loader to get batches from
        n_batches: Number of batches to get, if None get all

    Returns:
        List of batches
    """
    batches = []
    for i, batch in enumerate(loader):
        if n_batches and i >= n_batches:
            break
        batches.append(batch)
    return batches


def train_epoch(
    model: torch.nn.Module,
    train_batches: List,
    val_batches: List,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    cfg: OmegaConf,
    global_step: int,
    best_val_loss: float,
) -> Tuple[int, float]:
    """Train model for one epoch using fixed batches."""
    steps = 0

    print(f"Training epoch {epoch+1}")
    for batch in train_batches:
        x, y = batch[0], batch[1]  # ignore ID if present
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = model.calculate_loss(outputs, y)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Calculate metrics
        batch_metrics = model.calculate_metrics(outputs, y, loss)

        steps += 1
        global_step += 1

        # Log every n steps
        if steps % cfg.trainer.log_every_n_steps == 0:
            wandb.log(
                {
                    "train/step": global_step,
                    "train/step_loss": batch_metrics["loss"],
                    "train/step_f1": batch_metrics["f1"],
                    "train/step_precision": batch_metrics["precision"],
                    "train/step_recall": batch_metrics["recall"],
                    "train/learning_rate": (
                        scheduler.get_last_lr()[0] if scheduler else cfg.model.optim.lr
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
                    "val/step_precision": val_metrics["precision"],
                    "val/step_recall": val_metrics["recall"],
                },
                step=global_step,
            )

            model.train()

    # Reset metrics for next epoch
    model.reset_metrics()

    return global_step, best_val_loss


@hydra.main(config_path=SRC_PATH, config_name="config", version_base=None)
def main(cfg: OmegaConf) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration object
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
    train_loader, val_loader = load_dataloaders(cfg.data, fold=0)

    # Get fixed batches for training and validation
    train_batches = get_fixed_batches(train_loader, cfg.trainer.n_batches_train)
    val_batches = get_fixed_batches(val_loader, cfg.trainer.n_batches_val)

    # Print total batches
    print(f"Fixed training batches: {len(train_batches)}")
    if cfg.trainer.use_validation:
        print(f"Fixed validation batches: {len(val_batches)}")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.model.optim.lr,
        weight_decay=cfg.model.weight_decay,
    )

    scheduler = None
    if cfg.trainer.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=100  # warmup steps
        )

    # Training loop
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(cfg.trainer.epochs):
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
