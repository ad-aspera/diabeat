import torch
import torch.nn.functional as F
import lightning as L
from dataclasses import dataclass
from torch import nn
from omegaconf import OmegaConf


class HRVTransformer(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # Feature projection for raw HRV input
        self.feature_linear = nn.Linear(1, self.config.feature_dim)
        self.feature_ln = nn.LayerNorm(self.config.feature_dim)

        # Learned positional embedding
        self.pos_embedding = nn.Embedding(
            self.config.n_peaks_per_sample, self.config.feature_dim
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.feature_dim,
            nhead=self.config.n_heads,
            batch_first=True,
            dropout=self.config.dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.num_encoder_layers
        )
        self.transformer_ln = nn.LayerNorm(self.config.feature_dim)

        # Classification head for 3 classes
        self.classifier_head = nn.Linear(self.config.feature_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input shape is [batch_size, n_peaks_per_sample, 1]
        x = x.unsqueeze(-1)
        x = self.feature_linear(x)
        x = self.feature_ln(x)

        # Add positional embeddings
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions).expand(x.shape[0], -1, -1)

        # Transformer encoder
        x = self.transformer_encoder(x)
        x = self.transformer_ln(x)

        # Mean pooling
        x = torch.mean(x, dim=1)

        # Classification
        x = self.classifier_head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()  # Convert to long for CrossEntropy

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Calculate and log metrics
        precision, recall, f1 = self.calc_metrics(y_hat, y)
        self.log("train/loss", loss, batch_size=x.size(0), logger=True)
        self.log("train/precision", precision, batch_size=x.size(0), logger=True)
        self.log("train/recall", recall, batch_size=x.size(0), logger=True)
        self.log("train/f1", f1, batch_size=x.size(0), logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()  # Convert to long for CrossEntropy

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Calculate and log metrics
        precision, recall, f1 = self.calc_metrics(y_hat, y)
        self.log("val/loss", loss)
        self.log("val/precision", precision)
        self.log("val/recall", recall)
        self.log("val/f1", f1)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def calc_metrics(self, y_hat, y):
        y_pred = torch.argmax(y_hat, dim=1)

        # Calculate metrics for each class and average
        precisions = []
        recalls = []
        f1s = []

        for cls in range(3):
            tp = torch.sum((y_pred == cls) & (y == cls)).float()
            fp = torch.sum((y_pred == cls) & (y != cls)).float()
            fn = torch.sum((y_pred != cls) & (y == cls)).float()

            precision = torch.where(
                tp + fp > 0, tp / (tp + fp), torch.tensor(0.0, device=y.device)
            )
            recall = torch.where(
                tp + fn > 0, tp / (tp + fn), torch.tensor(0.0, device=y.device)
            )
            f1 = torch.where(
                precision + recall > 0,
                2 * (precision * recall) / (precision + recall),
                torch.tensor(0.0, device=y.device),
            )

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        # Return macro averages
        return (
            torch.mean(torch.stack(precisions)),
            torch.mean(torch.stack(recalls)),
            torch.mean(torch.stack(f1s)),
        )


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
