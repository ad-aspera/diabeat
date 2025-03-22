import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf
import lightning as L


class HRV_1DCNN(L.LightningModule):
    """1D CNN model for HRV analysis.

    A PyTorch Lightning implementation of a 1D CNN for analyzing heart rate variability signals.
    The model consists of multiple convolutional layers followed by fully connected layers
    for binary classification of heart rate variability data.

    Attributes:
        config (OmegaConf): Configuration object containing model hyperparameters
        input_norm (nn.BatchNorm1d): Batch normalization for input
        conv_layers (nn.ModuleList): List of convolutional layers
        pool (nn.MaxPool1d): Pooling layer
        activation (nn.LeakyReLU): Activation function
        dropout (nn.Dropout): Dropout layer for regularization
        fc1 (nn.Linear): First fully connected layer
        fc2 (nn.Linear): Output layer for binary classification
    """

    def __init__(self, config: OmegaConf) -> None:
        """Initialize the model with given configuration.

        Constructs the 1D CNN architecture with layers as specified in the config,
        including convolutional layers, pooling, activation, and fully connected layers.

        Args:
            config (OmegaConf): Configuration object containing model hyperparameters
                including channels, kernel sizes, stride, padding, dropout rate, etc.
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

        Processes the input HRV signal through convolutional layers, pooling,
        activation functions, and fully connected layers to produce binary
        classification logits.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
                or (batch_size, 1, sequence_length) containing HRV signal data

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,) containing binary
                classification logits (pre-sigmoid values)
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
        """Calculate precision, recall, and F1 score for binary classification.

        Computes classification metrics based on predicted logits and ground truth labels.
        Applies sigmoid activation and 0.5 threshold to convert logits to binary predictions.

        Args:
            y_hat (torch.Tensor): Predicted logits from the model
            y (torch.Tensor): Ground truth binary labels (0 or 1)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - precision: Precision score (TP / (TP + FP))
                - recall: Recall score (TP / (TP + FN))
                - F1: F1 score (2 * precision * recall / (precision + recall))
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
        """Execute a single training step.

        Performs forward pass, loss calculation, metric computation, and logging
        for a batch of training data.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Tuple containing:
                - x: Input HRV signals of shape (batch_size, sequence_length)
                - y: Binary target labels of shape (batch_size,)
            batch_idx (int): Index of the current batch

        Returns:
            torch.Tensor: Computed loss value for backpropagation
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
        """Configure the optimizer for training.

        Sets up the Adam optimizer with learning rate from configuration.

        Returns:
            dict: Dictionary containing the configured optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.optim.lr)
        return {"optimizer": optimizer}

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Execute a single validation step.

        Performs forward pass, loss calculation, metric computation, and logging
        for a batch of validation data.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Tuple containing:
                - x: Input HRV signals of shape (batch_size, sequence_length)
                - y: Binary target labels of shape (batch_size,)
            batch_idx (int): Index of the current batch

        Returns:
            torch.Tensor: Computed validation loss value
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
        """Calculate the total number of trainable parameters in the model.

        Returns:
            int: Number of parameters in the model
        """
        return sum(p.numel() for p in self.parameters())
