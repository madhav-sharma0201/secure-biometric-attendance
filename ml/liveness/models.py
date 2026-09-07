"""Liveness model architectures.

Two models, one shared backbone:

  LivenessCNN      single frame  -> backbone -> pool -> FC -> logit   (baseline / control)
  LivenessCNNLSTM  N frames      -> backbone -> BiLSTM -> FC -> logit (main model)

They deliberately share `build_backbone` so the ablation isolates one variable: whether
temporal modelling helps. If the two models used different feature extractors, a
difference in ACER would tell us nothing about the LSTM.

Why MobileNetV3-Small: it must train inside a free Kaggle session and run on a CPU in
the deployed container. ~2.5M params against ResNet-50's ~25M. Anti-spoofing cues are
mostly local texture (moire, print grain, screen reflection), which does not need a
deep high-capacity backbone; capacity is better spent on the temporal side.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


def build_backbone(name: str = "mobilenet_v3_small", pretrained: bool = True):
    """Returns (feature_extractor, feature_dim). Classifier head removed."""
    if name == "mobilenet_v3_small":
        weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.mobilenet_v3_small(weights=weights)
        feat_dim = m.classifier[0].in_features  # 576
        m.classifier = nn.Identity()
        return m, feat_dim

    if name == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.efficientnet_b0(weights=weights)
        feat_dim = m.classifier[1].in_features  # 1280
        m.classifier = nn.Identity()
        return m, feat_dim

    raise ValueError(f"unknown backbone {name!r}")


class LivenessCNN(nn.Module):
    """Single-frame baseline. Input (B, 3, H, W) -> (B,) logit.

    This is the control. If the temporal model cannot beat it, the extra complexity is
    not justified and we say so in the report.
    """

    def __init__(self, backbone="mobilenet_v3_small", pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone, pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


class LivenessCNNLSTM(nn.Module):
    """Temporal model. Input (B, N, 3, H, W) -> (B,) logit.

    The backbone is applied to all B*N frames in one batched pass rather than looping
    over the sequence: same result, far better GPU utilisation, and it keeps the
    weights genuinely shared across timesteps.
    """

    def __init__(
        self,
        backbone="mobilenet_v3_small",
        pretrained=True,
        temporal_model="lstm",
        hidden_size=128,
        bidirectional=True,
        dropout=0.3,
    ):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone, pretrained)

        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[temporal_model]
        self.rnn = rnn_cls(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n = x.shape[:2]
        feats = self.backbone(x.flatten(0, 1))       # (B*N, D)
        feats = feats.view(b, n, -1)                 # (B, N, D)
        seq, _ = self.rnn(feats)                     # (B, N, H*dirs)
        # Mean-pool over time rather than taking the last hidden state: spoof evidence
        # can appear in any frame, and pooling avoids over-weighting the final one.
        return self.head(seq.mean(dim=1)).squeeze(-1)


def build_model(cfg: dict) -> nn.Module:
    """Construct the model described by a config dict (configs/liveness.yaml -> model)."""
    arch = cfg.get("arch", "cnn_lstm")
    if arch == "cnn":
        return LivenessCNN(
            backbone=cfg.get("backbone", "mobilenet_v3_small"),
            pretrained=cfg.get("pretrained", True),
            dropout=cfg.get("dropout", 0.3),
        )
    if arch == "cnn_lstm":
        return LivenessCNNLSTM(
            backbone=cfg.get("backbone", "mobilenet_v3_small"),
            pretrained=cfg.get("pretrained", True),
            temporal_model=cfg.get("temporal_model", "lstm"),
            hidden_size=cfg.get("hidden_size", 128),
            bidirectional=cfg.get("bidirectional", True),
            dropout=cfg.get("dropout", 0.3),
        )
    raise ValueError(f"unknown arch {arch!r}")
