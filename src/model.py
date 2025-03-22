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
        self.n_outs = 3 if self.config.class_config == "all" else 1

        self.classifier_head = nn.Linear(self.config.feature_dim, self.n_outs)

        # Initialize metrics - will be created dynamically during calculation
        if self.config.use_class_weights:
            assert len(self.config.class_weights) == self.n_outs, (
                f"Number of class weights ({len(self.config.class_weights)}) must match "
                f"number of outputs ({self.n_outs})"
            )
            print(f"Using class weights: {self.config.class_weights}")
            self.class_weights = torch.tensor(self.config.class_weights)
        else:
            self.class_weights = None

    def _create_sinusoidal_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings for the transformer.

        Implements the fixed sinusoidal encoding described in the paper
        "Attention Is All You Need" for providing position information.

        Args:
            seq_len (int): Maximum sequence length
            d_model (int): Dimension of the model (feature dimension)

        Returns:
            torch.Tensor: Positional encodings tensor of shape (seq_len, d_model)
        """
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pos_encoding = torch.zeros(seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding

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
            # Binary classification with BCE loss
            if self.config.use_class_weights:
                class_weights = self.class_weights.to(outputs.device)
                pos_weight = class_weights  # Weight for positive class
                return F.binary_cross_entropy_with_logits(
                    outputs.squeeze(), targets.float(), pos_weight=pos_weight
                )
            return F.binary_cross_entropy_with_logits(
                outputs.squeeze(), targets.float()
            )
        else:
            # Multiclass with cross entropy
            if self.config.use_class_weights:
                class_weights = self.class_weights.to(outputs.device)
                return F.cross_entropy(outputs, targets.long(), weight=class_weights)
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
