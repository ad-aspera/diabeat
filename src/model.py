import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf
import math
from typing import Dict
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix,
    BinaryConfusionMatrix,
)
from torchmetrics import FBetaScore


class HRVTransformer(nn.Module):
    """Transformer-based model for heart rate variability (HRV) classification.

    This model uses a transformer encoder architecture with CLS token for classification
    of HRV signals. It supports both binary and multiclass classification.

    Attributes:
        config (OmegaConf): Configuration object containing model parameters
        feature_linear (nn.Linear): Linear projection for raw HRV input
        feature_ln (nn.LayerNorm): Layer normalization for feature embeddings
        pos_embedding (nn.Parameter or torch.Tensor): Positional encoding (learned or sinusoidal)
        transformer_encoder (nn.TransformerEncoder): Main transformer encoder layers
        transformer_ln (nn.LayerNorm): Layer normalization for transformer outputs
        cls_token (nn.Parameter): Learnable CLS token for classification
        n_outs (int): Number of output classes (1 for binary, 3 for multiclass)
        classifier_head (nn.Linear): Final classification layer
        class_weights (Optional[torch.Tensor]): Optional weights for weighted loss functions
    """

    def __init__(self, config: OmegaConf) -> None:
        """Initialize the HRV Transformer model.

        Args:
            config (OmegaConf): Configuration object containing model parameters
                such as feature dimensions, number of heads, layers, dropout, etc.
        """
        super().__init__()
        self.config = config

        # required for positional encoding
        self.dim_model = config.dim_model

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.dim_feedforward,
            batch_first=True,
            dropout=self.config.dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.num_encoder_layers
        )
        self.transformer_ln = nn.LayerNorm(self.dim_model)

        # Initialize CLS token with smaller values
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim_model) * 0.02)

        # Add layer normalization before classification
        self.n_outs = 3 if self.config.class_config == "all" else 1

        self.classifier_head = nn.Linear(self.dim_model, self.n_outs)

    def get_time_positional_encoding(
        self,
        time_indices: torch.Tensor | list[int],
        max_len: int = 135000,
    ) -> torch.Tensor:
        """Compute 1D positional encodings using sine and cosine functions.

        Args:
            d_model (int): Dimension of the model embeddings. Must be even.
            time_indices (torch.Tensor | list[int]): Time indices for which to compute encodings.
                Shape: (batch_size, N) where N is the number of positions per sample.

        Returns:
            torch.Tensor: Positional encodings matrix of shape (batch_size, N, d_model) where:
                - batch_size is the number of samples
                - N is the number of positions per sample
                - d_model is the embedding dimension

        Raises:
            ValueError: If d_model is odd
        """
        if self.dim_model % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(self.dim_model)
            )

        assert isinstance(time_indices, torch.Tensor), "time_indices must be a tensor"

        batch_size, seq_len = time_indices.shape
        pe = torch.zeros(
            batch_size, seq_len, self.dim_model, device=time_indices.device
        )
        position = time_indices.unsqueeze(-1)
        div_term = torch.exp(
            (
                torch.arange(
                    0, self.dim_model, 2, dtype=torch.float, device=time_indices.device
                )
                * -(math.log(max_len) / self.dim_model)
            )
        )
        pe[..., 0::2] = torch.sin(position.float() * div_term)
        pe[..., 1::2] = torch.cos(position.float() * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer model.

        Process input HRV signals through the transformer network to produce
        classification logits. This includes adding CLS token, positional encodings,
        and transformer encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
                or (batch_size, sequence_length, 1)

        Returns:
            torch.Tensor: Output logits tensor of shape (batch_size, n_outs)
                For binary classification (n_outs=1), a single logit per sample
                For multiclass (n_outs=3), logits for each class
        """
        # of shape (batch_size, n_peaks_per_sample, dim_model)
        pos_emb = self.get_time_positional_encoding(x)

        # Add CLS token to beginning of sequence
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        # x is of shape (batch_size, n_peaks_per_sample + 1, dim_model)
        x = torch.cat((cls_tokens, pos_emb), dim=1)

        # Transformer encoder
        x = self.transformer_encoder(x)
        x = self.transformer_ln(x)

        # Use CLS token output for classification
        x = x[:, 0]  # Take the first token (CLS token)
        x = self.classifier_head(x)
        return x

    def calculate_metrics(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate evaluation metrics for model predictions.

        Calculates classification metrics (F1, F2, precision, recall) for
        binary or multiclass classification without storing internal state.

        Args:
            outputs (torch.Tensor): Model output logits
            targets (torch.Tensor): Target class labels

        Returns:
            Dict[str, float]: Dictionary containing metrics:
                - f1: F1 score
                - f2: F2 score
                - precision: Precision score
                - recall: Recall score
        """
        device = outputs.device

        metrics = {}

        if self.n_outs == 1:
            # Binary classification metrics
            # Convert logits to probabilities and predictions
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).float()

            # Create metric instances for each calculation
            f1_score = BinaryF1Score().to(device)
            f2_score = FBetaScore(beta=2.0, task="binary").to(device)
            precision = BinaryPrecision().to(device)
            recall = BinaryRecall().to(device)

            metrics = {
                "f1": f1_score(preds, targets),
                "f2": f2_score(preds, targets),
                "precision": precision(preds, targets),
                "recall": recall(preds, targets),
            }
        else:
            # Multiclass metrics
            _, preds = torch.max(outputs, 1)

            # Create metric instances for each calculation
            f1_score = MulticlassF1Score(num_classes=self.n_outs).to(device)
            f2_score = FBetaScore(
                beta=2.0, task="multiclass", num_classes=self.n_outs
            ).to(device)
            precision = MulticlassPrecision(num_classes=self.n_outs).to(device)
            recall = MulticlassRecall(num_classes=self.n_outs).to(device)

            metrics = {
                "f1": f1_score(preds, targets),
                "f2": f2_score(preds, targets),
                "precision": precision(preds, targets),
                "recall": recall(preds, targets),
            }

        return metrics

    def get_confusion_matrix(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the confusion matrix for model predictions.

        Calculates a confusion matrix comparing predictions to targets for
        either binary or multiclass classification.

        For binary classification (n_outs=1), the confusion matrix is a 2x2 matrix with structure:
        [[TN, FP],
         [FN, TP]]

        where:
        - TN: Controls (class 0) correctly classified as Controls
        - FP: Controls (class 0) incorrectly classified as DPN
        - FN: DPN (class 1) incorrectly classified as Controls
        - TP: DPN (class 1) correctly classified as DPN

        Args:
            outputs (torch.Tensor): Model output logits
            targets (torch.Tensor): Target class labels

        Returns:
            torch.Tensor: Confusion matrix as a tensor:
                - For binary: 2x2 confusion matrix
                - For multiclass: NxN confusion matrix where N is the number of classes
        """
        device = outputs.device

        if self.n_outs == 1:
            # Binary confusion matrix
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).float()
            cm = BinaryConfusionMatrix().to(device)
        else:
            # Multiclass confusion matrix
            _, preds = torch.max(outputs, 1)
            cm = MulticlassConfusionMatrix(num_classes=self.n_outs).to(device)

        return cm(preds, targets)

    def calculate_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate appropriate loss based on the classification task.

        Calculates either binary cross-entropy loss or multiclass cross-entropy loss
        depending on the model configuration, with optional class weights.

        Args:
            outputs (torch.Tensor): Model output logits
            targets (torch.Tensor): Target class labels

        Returns:
            torch.Tensor: Loss value as a tensor
        """
        if self.n_outs == 1:
            return F.binary_cross_entropy_with_logits(
                outputs.squeeze(), targets.float()
            )
        else:
            return F.cross_entropy(outputs, targets.long())


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across random, numpy, and PyTorch.

    Args:
        seed (int, optional): Random seed to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def num_params(model: nn.Module) -> int:
    """Calculate the total number of parameters in a PyTorch model.

    Args:
        model (nn.Module): PyTorch model to analyze

    Returns:
        int: Total number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters())


def move_batch_to_device(
    batch: list[torch.Tensor], device: torch.device
) -> list[torch.Tensor]:
    """Move all tensors in a batch to a specified device.

    Args:
        batch (list[torch.Tensor]): Batch of tensors to move
        device (torch.device): Device to move tensors to

    Returns:
        list[torch.Tensor]: Batch with tensors moved to the specified device
    """
    batch = [item.to(device) for item in batch if isinstance(item, torch.Tensor)]
    return batch
