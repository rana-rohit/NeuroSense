"""
deep_model.py
Responsible for: Deep learning models for EEG+ECG emotion recognition.

Models:
  - EEGNet       : Compact CNN designed for EEG (Lawhern et al., 2018)
  - CNN1D        : 1D CNN for parallel EEG+ECG feature extraction
  - CNNLSTM      : CNN + bidirectional LSTM for temporal modelling
  - FusionModel  : Late-fusion of EEG + ECG branches → classifier head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ── Shared utilities ──────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv1d → BatchNorm → ReLU block."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 stride: int = 1, padding: int = 0, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel,
                      stride=stride, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Model 1: EEGNet ───────────────────────────────────────────────────────────

class EEGNet(nn.Module):
    """
    EEGNet — compact, general-purpose EEG classification CNN.
    Reference: Lawhern et al. (2018), arXiv:1611.08024

    Input : (batch, 14, eeg_samples)   — channels × time
    Output: (batch, n_classes)

    Args:
        n_channels  : number of EEG channels (14 for DREAMER)
        n_samples   : number of time samples per window
        n_classes   : 2 for binary classification
        fs          : sampling rate (128 Hz)
        F1          : temporal filter count
        D           : depth multiplier
        F2          : pointwise filter count
        dropout     : dropout probability
    """

    def __init__(self, n_channels: int = 14, n_samples: int = 512,
                 n_classes: int = 2, fs: int = 128,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 dropout: float = 0.5):
        super().__init__()

        # Block 1: Temporal conv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, fs // 2), padding=(0, fs // 4), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Block 2: Depthwise spatial conv
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Block 3: Separable conv
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out   = self.block3(self.block2(self.block1(dummy)))
            flat  = out.view(1, -1).shape[1]

        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time) or (batch, time, channels)
        """
        if x.dim() == 3 and x.shape[-1] != x.shape[-2]:
            # Ensure (batch, channels, time)
            if x.shape[1] > x.shape[2]:
                x = x.transpose(1, 2)
        x = x.unsqueeze(1)                    # → (batch, 1, channels, time)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Model 2: 1D CNN branch ────────────────────────────────────────────────────

class CNN1DBranch(nn.Module):
    """
    Lightweight 1D CNN for a single modality (EEG or ECG).

    Input : (batch, channels, time)
    Output: (batch, out_dim)
    """

    def __init__(self, in_channels: int, out_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnRelu(in_channels, 32, kernel=7, padding=3),
            nn.MaxPool1d(2),

            ConvBnRelu(32, 64, kernel=5, padding=2),
            nn.MaxPool1d(2),

            ConvBnRelu(64, 128, kernel=3, padding=1),
            nn.MaxPool1d(2),

            ConvBnRelu(128, 128, kernel=3, padding=1),
            nn.AdaptiveAvgPool1d(1),          # → (batch, 128, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


# ── Model 3: CNN-LSTM ─────────────────────────────────────────────────────────

class CNNLSTMBranch(nn.Module):
    """
    CNN feature extraction → Bidirectional LSTM temporal modelling.

    Input : (batch, channels, time)
    Output: (batch, out_dim)
    """

    def __init__(self, in_channels: int, out_dim: int = 128,
                 lstm_hidden: int = 64, lstm_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            ConvBnRelu(in_channels, 32, kernel=7, padding=3),
            nn.MaxPool1d(2),
            ConvBnRelu(32, 64, kernel=5, padding=2),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        x = self.cnn(x)                        # (batch, 64, T')
        x = x.permute(0, 2, 1)                 # (batch, T', 64)
        out, _ = self.lstm(x)
        x = out[:, -1, :]                       # last timestep
        return self.head(x)


# ── Model 4: Fusion Model ─────────────────────────────────────────────────────

class FusionModel(nn.Module):
    """
    Late-fusion model: independent EEG + ECG branches → shared classifier.

    Supports two branch types:
        'cnn'     → CNN1DBranch
        'cnnlstm' → CNNLSTMBranch

    Input:
        eeg : (batch, eeg_time, 14)
        ecg : (batch, ecg_time, 2)

    Output: (batch, n_classes)
    """

    def __init__(
        self,
        n_eeg_channels : int   = 14,
        n_ecg_channels : int   = 2,
        branch_type    : str   = "cnnlstm",
        branch_dim     : int   = 128,
        n_classes      : int   = 2,
        dropout        : float = 0.4,
        modality       : str   = "fusion",
    ):
        super().__init__()
        self.modality = modality

        branch_cls = CNN1DBranch if branch_type == "cnn" else CNNLSTMBranch

        self.eeg_branch = branch_cls(n_eeg_channels, out_dim=branch_dim,
                                     dropout=dropout)
        self.ecg_branch = branch_cls(n_ecg_channels, out_dim=branch_dim,
                                     dropout=dropout)

        # Attention gate: learn relative importance of EEG vs ECG
        self.attention = nn.Sequential(
            nn.Linear(branch_dim * 2, 2),
            nn.Softmax(dim=-1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(branch_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(64, n_classes),
        )

    def forward(self, eeg: torch.Tensor,
                ecg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg: (batch, eeg_time, 14)  — time-first from DataLoader
            ecg: (batch, ecg_time, 2)
        """
        # Transpose to (batch, channels, time) for Conv1d
        eeg = eeg.transpose(1, 2)          # (batch, 14, eeg_time)
        ecg = ecg.transpose(1, 2)          # (batch, 2,  ecg_time)

        feat_eeg = self.eeg_branch(eeg)    # (batch, branch_dim)
        feat_ecg = self.ecg_branch(ecg)    # (batch, branch_dim)

        # Modality ablation: zero out unused branch features
        if self.modality == "eeg":
            feat_ecg = torch.zeros_like(feat_ecg)
        elif self.modality == "ecg":
            feat_eeg = torch.zeros_like(feat_eeg)

        fused = torch.cat([feat_eeg, feat_ecg], dim=-1)   # (batch, branch_dim*2)

        # Attention weighting
        attn   = self.attention(fused)                     # (batch, 2)
        w_eeg  = attn[:, 0:1]
        w_ecg  = attn[:, 1:2]
        fused  = torch.cat([
            feat_eeg * w_eeg,
            feat_ecg * w_ecg,
        ], dim=-1)

        return self.classifier(fused)


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(model_type: str, config: dict, modality: str = "fusion") -> nn.Module:
    """
    Factory function — instantiate model from config.

    Args:
        model_type: 'eegnet' | 'cnn' | 'cnnlstm' | 'fusion'
        modality  : 'eeg' | 'ecg' | 'fusion' (for ablation)
        config    : loaded YAML config dict

    Returns:
        nn.Module
    """
    mc = config.get("model", {})
    tc = config.get("training", {})

    n_classes = 2
    dropout   = float(mc.get("dropout", 0.4))
    eeg_samples = int(
        config["data"]["segment_length"] * config["data"]["sampling_rate_eeg"]
    )

    if model_type == "eegnet":
        return EEGNet(
            n_channels=14,
            n_samples=eeg_samples,
            n_classes=n_classes,
            fs=config["data"]["sampling_rate_eeg"],
            dropout=dropout,
        )

    elif model_type in ("cnn", "cnnlstm"):
        return FusionModel(
            n_eeg_channels=14,
            n_ecg_channels=2,
            branch_type=model_type,
            branch_dim=int(mc.get("branch_dim", 128)),
            n_classes=n_classes,
            dropout=dropout,
            modality=modality,
        )

    elif model_type == "fusion":
        return FusionModel(
            n_eeg_channels=14,
            n_ecg_channels=2,
            branch_type="cnnlstm",
            branch_dim=int(mc.get("branch_dim", 128)),
            n_classes=n_classes,
            dropout=dropout,
            modality=modality,
        )

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose: eegnet | cnn | cnnlstm | fusion"
        )