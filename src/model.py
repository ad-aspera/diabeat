import torch
import torch.nn.functional as F
import lightning as L
from dataclasses import dataclass
from torch import nn
from omegaconf import OmegaConf


class HRV_1DCNN(L.LightningModule):
    """1D CNN model for HRV analysis.

    A PyTorch Lightning implementation of a 1D CNN for analyzing heart rate variability signals.
    The model consists of multiple convolutional layers followed by fully connected layers
    for binary classification.
    """

    def __init__(self, config: OmegaConf) -> None:
        """Initialize the model with given configuration.

        Args:
            config: OmegaConf object containing model hyperparameters
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        # Input normalization layer
        self.input_norm = nn.BatchNorm1d(self.config.channels[0])

        # Create dynamic convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(len(self.config.channels) - 1):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=self.config.channels[i],
                    out_channels=self.config.channels[i + 1],
                    kernel_size=self.config.kernels[i],
                    stride=self.config.stride,
                    padding=self.config.padding,
                )
            )

        # Pooling and activation
        self.pool = nn.MaxPool1d(kernel_size=self.config.pool_size)
        self.activation = nn.LeakyReLU(negative_slope=self.config.leaky_relu_slope)
        self.dropout = nn.Dropout(self.config.dropout)

        # Calculate the size after convolutions and pooling
        final_length = self.config.n_peaks_per_sample // (
            2 ** (len(self.conv_layers))
        )  # Dynamic pooling calculation
        self.fc1 = nn.Linear(
            self.config.channels[-1] * final_length, self.config.fc_dims[0]
        )
        self.fc2 = nn.Linear(self.config.fc_dims[0], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length) or (batch_size, 1, sequence_length)

        Returns:
            Tensor of shape (batch_size,) containing logits
        """
        # Ensure input is in the right shape
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Apply normalization
        x = self.input_norm(x)

        # Dynamic convolutional blocks
        for conv in self.conv_layers:
            x = self.activation(conv(x))
            x = self.pool(x)
            x = self.dropout(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def calc_metrics(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate precision, recall, and F1 score.

        Args:
            y_hat: Predicted logits
            y: Ground truth labels

        Returns:
            Tuple of (precision, recall, F1) scores
        """
        # Apply sigmoid first, then threshold at 0.5
        y_pred = (torch.sigmoid(y_hat) >= 0.5).float()

        # Calculate true positives, false positives, false negatives
        tp = torch.sum((y_pred == 1) & (y == 1)).float()
        fp = torch.sum((y_pred == 1) & (y == 0)).float()
        fn = torch.sum((y_pred == 0) & (y == 1)).float()

        # Calculate precision, recall, and F1
        precision = tp / (tp + fp + self.config.eps)
        recall = tp / (tp + fn + self.config.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.config.eps)

        return precision, recall, f1

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Tuple of (inputs, labels)
            batch_idx: Index of current batch

        Returns:
            Loss tensor
        """
        x, y = batch
        y_hat = self(x).view(-1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        # Calculate and log metrics
        precision, recall, f1 = self.calc_metrics(y_hat, y)
        self.log("train/loss", loss)
        self.log("train/precision", precision)
        self.log("train/recall", recall)
        self.log("train/f1", f1)

        return loss

    def configure_optimizers(self) -> dict:
        """Configure optimizer.

        Returns:
            Dictionary containing optimizer configuration
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.optim.lr)
        return {"optimizer": optimizer}

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Tuple of (inputs, labels)
            batch_idx: Index of current batch

        Returns:
            Loss tensor
        """
        x, y = batch
        y_hat = self(x).view(-1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        # Calculate and log metrics
        precision, recall, f1 = self.calc_metrics(y_hat, y)
        self.log("val/loss", loss)
        self.log("val/precision", precision)
        self.log("val/recall", recall)
        self.log("val/f1", f1)

        return loss

    def num_params(self) -> int:
        """Calculate the number of parameters in the model.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters())
