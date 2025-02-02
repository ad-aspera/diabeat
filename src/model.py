import torch
import torch.nn.functional as F
import lightning as L
from dataclasses import dataclass
from torch import nn


@dataclass
class HRV_1DCNN_Config:
    """Configuration class for the 1D CNN model for HRV analysis.

    Attributes:
        channels: List of channel dimensions for each layer [input, conv1, conv2, conv3]
        kernels: List of kernel sizes for each convolutional layer
        fc_dims: List of dimensions for fully connected layers
        input_length: Length of input sequence
        learning_rate: Learning rate for optimizer
        dropout_rate: Dropout probability
    """

    # Model architecture
    # Number of channels at each layer
    channels: list[int] = [1, 32, 64, 128]  # [input, conv1, conv2, conv3]

    # Kernel sizes for each conv layer
    kernels: list[int] = [15, 31, 61]  # [conv1, conv2, conv3]

    # Fully connected layer dimensions
    fc_dims: list[int] = [64]

    # Input sequence length
    input_length: int = 600

    # Training hyperparameters
    learning_rate: float = 1e-3
    dropout_rate: float = 0.5


class HRV_1DCNN(L.LightningModule):
    """1D CNN model for HRV analysis.

    A PyTorch Lightning implementation of a 1D CNN for analyzing heart rate variability signals.
    The model consists of multiple convolutional layers followed by fully connected layers
    for binary classification.
    """

    def __init__(self, config: HRV_1DCNN_Config) -> None:
        """Initialize the model with given configuration.

        Args:
            config: Configuration object containing model hyperparameters
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()

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
                    stride=1,
                    padding="same",
                )
            )

        # Pooling and activation
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.config.dropout_rate)

        # Calculate the size after convolutions and pooling
        final_length = self.config.input_length // (
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
        precision = tp / (tp + fp + 1e-8)  # Add small epsilon to avoid division by zero
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

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
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/precision", precision, prog_bar=True)
        self.log("train/recall", recall, prog_bar=True)
        self.log("train/f1", f1, prog_bar=True)

        return loss

    def configure_optimizers(self) -> dict:
        """Configure optimizer.

        Returns:
            Dictionary containing optimizer configuration
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
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
