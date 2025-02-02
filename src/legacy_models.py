import torch
import torch.nn.functional as F
import lightning as L
from torch import nn


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
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        # Calculate and log metrics
        precision, recall, f1 = self.calc_metrics(y_hat, y)
        self.log("train/loss", loss, batch_size=x.size(0), logger=True)
        self.log("train/precision", precision, batch_size=x.size(0), logger=True)
        self.log("train/recall", recall, batch_size=x.size(0), logger=True)
        self.log("train/f1", f1, batch_size=x.size(0), logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def calc_metrics(self, y_hat, y):
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

    def validation_step(self, batch, batch_idx):
        x, y, id_ = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        # Calculate and log metrics
        precision, recall, f1 = self.calc_metrics(y_hat, y)
        self.log("val/loss", loss)
        self.log("val/precision", precision)
        self.log("val/recall", recall)
        self.log("val/f1", f1)

        return loss
