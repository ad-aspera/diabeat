import os
import pandas as pd
import torch
from torch import nn
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
import lightning as L
from dataclasses import dataclass
import math
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC


@dataclass
class ECGTransfomerConfig:
    num_leads: int = 12
    feature_dim: int = 64
    num_encoder_layers: int = 4
    n_heads: int = 8
    seq_len: int = 5000
    dropout: float = 0.1


class ECGTransformer(L.LightningModule):
    def __init__(self, config: ECGTransfomerConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()  # Save config for model checkpointing

        # Feature projection
        self.feature_linear = nn.Linear(self.config.num_leads, self.config.feature_dim)
        self.feature_ln = nn.LayerNorm(self.config.feature_dim)

        # Learned positional embedding
        self.pos_embedding = nn.Embedding(self.config.seq_len, self.config.feature_dim)

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

        # Classification head
        self.classifier_head = nn.Linear(self.config.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, num_leads]
        x = self.feature_linear(x)  # [batch_size, seq_len, feature_dim]
        x = self.feature_ln(x)

        # Add positional embedding
        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_embedding(positions)

        # Transformer encoder
        x = self.transformer_encoder(x)
        x = self.transformer_ln(x)

        # Global average pooling over sequence length
        x = x.mean(dim=1)  # [batch_size, feature_dim]

        # Classification
        x = self.classifier_head(x)  # [batch_size, 1]
        return x.squeeze(-1)  # [batch_size]

    def training_step(self, batch, batch_idx):
        x, y, id_ = batch
        y_hat = self(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train/loss", loss, batch_size=x.size(0), logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y, id_ = batch
        y_hat = self(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        self.log("val/loss", loss)
        return loss


class HRVClassifier(L.LightningModule):
    def __init__(self, learning_rate: float = 1e-3, dropout_rate: float = 0.5):
        super().__init__()
        self.save_hyperparameters()

        # Input normalization layer (normalize across the time dimension)
        self.input_norm = nn.BatchNorm1d(1)  # 1 channel

        # Convolutional layers - adjusted to use odd kernel sizes
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=15, stride=1, padding="same"
        )
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=31, stride=1, padding="same"
        )
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=61, stride=1, padding="same"
        )

        # Pooling and activation
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate the size after convolutions and pooling
        final_length = 600 // (2 * 2 * 2)  # After 3 pooling operations
        self.fc1 = nn.Linear(128 * final_length, 64)
        self.fc2 = nn.Linear(64, 1)  # Assuming binary classification

    def forward(self, x):
        # Ensure input is in the right shape (batch_size, 1, 600)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Apply normalization
        x = self.input_norm(x)

        # Convolutional blocks
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.activation(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(-1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
