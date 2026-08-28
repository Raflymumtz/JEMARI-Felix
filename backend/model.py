"""CNN + Transformer (self-attention) spatio-temporal model with an
auxiliary hand-landmark stream.

Follows the architecture described in the research proposal (BAB 3.3):
  - a MobileNet backbone extracts a spatial feature vector per frame,
  - a Transformer encoder (multi-head self-attention) models the temporal
    relationship between frames in a short window, capturing how the hand
    gesture evolves over time instead of relying on a single static frame,
  - the pooled temporal representation is classified into one of the
    BISINDO letter classes.

Two refinements over the first implementation:

  * **MobileNetV3-Small instead of MobileNetV2.** Roughly a third of the
    parameters for comparable ImageNet accuracy, which is what makes the
    exported model small enough to load quickly on a phone browser.

  * **A landmark branch.** MediaPipe already computes 21 hand joints in
    order to find the crop box, so their positions are free at inference
    time. For fingerspelling — where letters differ by finger placement
    rather than by colour or texture — joint geometry is a far more direct
    signal than pixels, and it stays reliable when lighting or skin tone
    drifts away from the training set. The CNN and landmark features are
    fused into a single token per frame; the Transformer above is unchanged.

    Frames where detection failed carry an all-zero landmark vector with a 0
    validity flag (see hand_crop.BLIND_LANDMARKS), which lets the model fall
    back to the CNN alone rather than trusting a bogus pose.
"""
import math

import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V2_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v2,
    mobilenet_v3_small,
)

import config

# backbone name -> (constructor, weights enum, output channels of .features)
BACKBONES = {
    "mobilenet_v3_small": (mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1, 576),
    "mobilenet_v2": (mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V1, 1280),
}


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
        backbone: str = None,
        d_model: int = config.D_MODEL,
        n_heads: int = config.N_HEADS,
        n_layers: int = config.N_LAYERS,
        ff_dim: int = config.FF_DIM,
        dropout: float = config.DROPOUT,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        backbone = backbone or config.BACKBONE
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}, expected one of {list(BACKBONES)}")
        builder, weights_enum, feature_dim = BACKBONES[backbone]
        self.backbone_name = backbone

        net = builder(weights=weights_enum if pretrained else None)
        self.cnn = net.features  # spatial feature extractor
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for p in self.cnn.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(feature_dim, config.D_CNN)

        # landmark branch: 21 joints (x, y, z) + validity flag -> compact vector
        self.landmark_encoder = nn.Sequential(
            nn.Linear(config.LANDMARK_DIM, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, config.D_LM),
            nn.LayerNorm(config.D_LM),
            nn.GELU(),
        )

        # fuse the two streams into one token per frame
        self.fuse = nn.Sequential(
            nn.Linear(config.D_CNN + config.D_LM, d_model),
            nn.LayerNorm(d_model),
        )

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

    # -- shared by training and the two-stage inference path ---------------

    def encode_frames(self, x, landmarks):
        """(B, T, C, H, W) + (B, T, LANDMARK_DIM) -> (B, T, d_model) tokens."""
        b, t = x.shape[:2]
        feats = self.cnn(x.flatten(0, 1))          # (B*T, feature_dim, H', W')
        feats = self.pool(feats).flatten(1)        # (B*T, feature_dim)
        feats = self.proj(feats)                   # (B*T, D_CNN)

        lm = self.landmark_encoder(landmarks.flatten(0, 1))  # (B*T, D_LM)

        tokens = self.fuse(torch.cat([feats, lm], dim=1))    # (B*T, d_model)
        return tokens.view(b, t, -1)

    def classify_tokens(self, tokens):
        """(B, T, d_model) -> (B, num_classes)."""
        encoded = self.temporal_encoder(self.pos_encoding(tokens))
        pooled = self.dropout(encoded.mean(dim=1))  # temporal average pooling
        return self.classifier(pooled)

    def forward(self, x, landmarks):
        return self.classify_tokens(self.encode_frames(x, landmarks))

    # -- real-time server path ---------------------------------------------

    @torch.no_grad()
    def embed_frame(self, frame_tensor, landmark_tensor):
        """Encode a single frame (C,H,W) + its landmarks into one d_model token.

        Used by the real-time server so each incoming frame is only run
        through the CNN once, then buffered for the Transformer pass.
        """
        self.eval()
        tokens = self.encode_frames(
            frame_tensor.unsqueeze(0).unsqueeze(0),
            landmark_tensor.unsqueeze(0).unsqueeze(0),
        )
        return tokens[0, 0]

    @torch.no_grad()
    def classify_embeddings(self, embeddings):
        """embeddings: (T, d_model) already-fused per-frame tokens."""
        self.eval()
        return self.classify_tokens(embeddings.unsqueeze(0)).squeeze(0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for name in BACKBONES:
        m = SignCNNTransformer(num_classes=26, backbone=name, pretrained=False)
        frames = torch.randn(2, config.WINDOW, 3, config.IMG_SIZE, config.IMG_SIZE)
        lm = torch.randn(2, config.WINDOW, config.LANDMARK_DIM)
        out = m(frames, lm)
        n = count_parameters(m)
        print(f"{name:<20} params={n/1e6:.2f}M  fp32={n*4/1e6:.1f}MB  out={tuple(out.shape)}")
