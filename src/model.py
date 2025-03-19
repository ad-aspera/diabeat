import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf
import math
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class HRVTransformer(nn.Module):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__()
        self.config = config

        # Feature projection for raw HRV input
        self.feature_linear = nn.Linear(1, self.config.feature_dim)
        self.feature_ln = nn.LayerNorm(self.config.feature_dim)

        # Positional encoding setup
        assert self.config.pos_encoding_type in ["learned", "sinusoidal"]
        if self.config.pos_encoding_type == "learned":
            self.pos_embedding = nn.Embedding(
                self.config.n_peaks_per_sample + 1, self.config.feature_dim
            )
        else:  # sinusoidal
            self.register_buffer(
                "pos_embedding",
                self._create_sinusoidal_encoding(
                    self.config.n_peaks_per_sample + 1, self.config.feature_dim
                ),
            )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.feature_dim,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.dim_feedforward,
            batch_first=True,
            dropout=self.config.dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.num_encoder_layers
        )
        self.transformer_ln = nn.LayerNorm(self.config.feature_dim)

        # Initialize CLS token with smaller values
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.feature_dim) * 0.02)

        # Add layer normalization before classification
        self.n_outs = 3 if self.config.class_config == "all" else 2
        assert len(self.config.class_weights) == self.n_outs, (
            f"Number of class weights ({len(self.config.class_weights)}) must match "
            f"number of outputs ({self.n_outs})"
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(self.config.feature_dim, self.config.feature_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.feature_dim, self.n_outs),
        )

        # Initialize metrics
        self.f1_metric = MulticlassF1Score(num_classes=self.n_outs)
        self.precision_metric = MulticlassPrecision(num_classes=self.n_outs)
        self.recall_metric = MulticlassRecall(num_classes=self.n_outs)
        self.class_weights = torch.tensor(self.config.class_weights)

    def _create_sinusoidal_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pos_encoding = torch.zeros(seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input shape is [batch_size, n_peaks_per_sample, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.feature_linear(x)
        x = self.feature_ln(x)

        # Add CLS token to beginning of sequence
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        if self.config.pos_encoding_type == "learned":
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            pos_emb = self.pos_embedding(positions)
        else:  # sinusoidal
            pos_emb = self.pos_embedding[: x.size(1)].unsqueeze(0)

        x = x + pos_emb.expand(x.shape[0], -1, -1)

        # Transformer encoder
        x = self.transformer_encoder(x)
        x = self.transformer_ln(x)

        # Use CLS token output for classification
        x = x[:, 0]  # Take the first token (CLS token)
        x = self.classifier_head(x)
        return x

    def calculate_metrics(self, outputs, targets, loss=None):
        """Calculate all metrics for the current batch/epoch."""
        # Move metrics to correct device if needed
        device = outputs.device
        self.f1_metric = self.f1_metric.to(device)
        self.precision_metric = self.precision_metric.to(device)
        self.recall_metric = self.recall_metric.to(device)

        # Get predictions
        _, predicted = torch.max(outputs.data, 1)

        # Update and compute metrics
        f1 = self.f1_metric(predicted, targets)
        precision = self.precision_metric(predicted, targets)
        recall = self.recall_metric(predicted, targets)

        metrics = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

        if loss is not None:
            metrics["loss"] = loss.item()

        return metrics

    def reset_metrics(self):
        """Reset all metrics at the end of epoch."""
        self.f1_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()

    def calculate_loss(self, outputs, targets):
        """Calculate loss with optional class weights."""
        class_weights = self.class_weights.to(outputs.device)
        return F.cross_entropy(outputs, targets.long(), weight=class_weights)


def set_seed(seed: int = 42):
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def num_params(model: nn.Module) -> int:
    """Calculate the number of parameters in the model.

    Args:
        model (nn.Module): The model to calculate the number of parameters for.

    Returns:
        int: The number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters())
