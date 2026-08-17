"""CNN (MobileNetV2) + Transformer (self-attention) spatio-temporal model.

Matches the architecture described in the research proposal (BAB 3.3):
  - MobileNetV2 backbone extracts a spatial feature vector per frame.
  - A Transformer encoder (multi-head self-attention) models the temporal
    relationship between frames in a short window, capturing how the hand
    gesture evolves over time instead of relying on a single static frame.
  - The pooled temporal representation is classified into one of the
    BISINDO letter classes.
"""
import math

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

import config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SignCNNTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_model: int = config.D_MODEL,
        n_heads: int = config.N_HEADS,
        n_layers: int = config.N_LAYERS,
        ff_dim: int = config.FF_DIM,
        dropout: float = config.DROPOUT,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        self.cnn = backbone.features  # spatial feature extractor
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for p in self.cnn.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(config.CNN_FEATURE_DIM, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=config.WINDOW + 4)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.cnn(x)              # (B*T, 1280, H', W')
        feats = self.pool(feats).flatten(1)  # (B*T, 1280)
        feats = self.proj(feats)         # (B*T, d_model)
        feats = feats.view(b, t, -1)     # (B, T, d_model)

        feats = self.pos_encoding(feats)
        encoded = self.temporal_encoder(feats)  # (B, T, d_model)

        pooled = encoded.mean(dim=1)     # temporal average pooling
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    @torch.no_grad()
    def embed_frame(self, frame_tensor):
        """Encode a single frame (C,H,W) into its projected d_model embedding.
        Used by the real-time server so each incoming frame is only run
        through the CNN once, then buffered for the Transformer pass.
        """
        self.eval()
        x = frame_tensor.unsqueeze(0)
        feats = self.cnn(x)
        feats = self.pool(feats).flatten(1)
        feats = self.proj(feats)
        return feats.squeeze(0)

    @torch.no_grad()
    def classify_embeddings(self, embeddings):
        """embeddings: (T, d_model) already-projected per-frame features."""
        self.eval()
        x = embeddings.unsqueeze(0)
        x = self.pos_encoding(x)
        encoded = self.temporal_encoder(x)
        pooled = encoded.mean(dim=1)
        logits = self.classifier(pooled)
        return logits.squeeze(0)
