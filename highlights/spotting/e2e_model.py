"""CPU-friendly re-implementation of the E2E-Spot model (Hong et al., ECCV 2022).

Module names mirror https://github.com/jhong93/spot so the released
`soccer_challenge_rny008gsm_gru_rgb` checkpoint loads with strict=True.
GSM code adapted from https://github.com/swathikirans/GSM (BSD-2).
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_GRU_HIDDEN_DIM = 768

SOCCERNET_CLASSES = [
    "Ball out of play", "Clearance", "Corner", "Direct free-kick", "Foul",
    "Goal", "Indirect free-kick", "Kick-off", "Offside", "Penalty", "Red card",
    "Shots off target", "Shots on target", "Substitution", "Throw-in",
    "Yellow card", "Yellow->red card",
]


class _GSM(nn.Module):
    def __init__(self, fPlane: int, num_segments: int):
        super().__init__()
        self.conv3D = nn.Conv3d(fPlane, 2, (3, 3, 3), stride=1, padding=(1, 1, 1), groups=2)
        self.tanh = nn.Tanh()
        self.fPlane = fPlane
        self.num_segments = num_segments
        self.bn = nn.BatchNorm3d(num_features=fPlane)
        self.relu = nn.ReLU()

    @staticmethod
    def _lshift(x: torch.Tensor) -> torch.Tensor:
        pad = x.new_zeros(x.size(0), x.size(1), 1, x.size(3), x.size(4))
        return torch.cat((x[:, :, 1:], pad), dim=2)

    @staticmethod
    def _rshift(x: torch.Tensor) -> torch.Tensor:
        pad = x.new_zeros(x.size(0), x.size(1), 1, x.size(3), x.size(4))
        return torch.cat((pad, x[:, :, :-1]), dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0) // self.num_segments
        shape = x.size(1), x.size(2), x.size(3)
        x = x.view(batch, self.num_segments, *shape).permute(0, 2, 1, 3, 4).contiguous()
        gate = self.tanh(self.conv3D(self.relu(self.bn(x))))
        g1, g2 = gate[:, 0].unsqueeze(1), gate[:, 1].unsqueeze(1)
        x1, x2 = x[:, : self.fPlane // 2], x[:, self.fPlane // 2 :]
        y1, y2 = g1 * x1, g2 * x2
        y1 = self._lshift(y1) + (x1 - y1)
        y2 = self._rshift(y2) + (x2 - y2)
        y1 = y1.view(batch, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)
        y2 = y2.view(batch, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)
        y = torch.cat(
            (
                y1.contiguous().view(batch, self.fPlane // 2, self.num_segments, *shape[1:]),
                y2.contiguous().view(batch, self.fPlane // 2, self.num_segments, *shape[1:]),
            ),
            dim=1,
        )
        return y.permute(0, 2, 1, 3, 4).contiguous().view(batch * self.num_segments, *shape)


class GatedShift(nn.Module):
    def __init__(self, net: nn.Module, n_segment: int, n_div: int = 4):
        super().__init__()
        channels = net.conv.in_channels
        self.fold_dim = math.ceil(channels // n_div / 4) * 4
        self.gsm = _GSM(self.fold_dim, n_segment)
        self.net = net
        self.n_segment = n_segment

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        y[:, : self.fold_dim] = self.gsm(x[:, : self.fold_dim])
        y[:, self.fold_dim :] = x[:, self.fold_dim :]
        return self.net(y)


class FCPrediction(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fc_out(x)


class GRUPrediction(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, hidden_dim: int):
        super().__init__()
        self._gru = nn.GRU(feat_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self._fc_out = FCPrediction(2 * hidden_dim, num_classes)
        self._dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class E2ESpot(nn.Module):
    """RegNetY + GSM feature extractor followed by a bidirectional GRU head."""

    def __init__(self, num_classes: int, feature_arch: str = "rny008_gsm", clip_len: int = 100):
        super().__init__()
        timm_name = {"rny002": "regnety_002", "rny008": "regnety_008"}[feature_arch.rsplit("_", 1)[0]]
        features = timm.create_model(timm_name, pretrained=False)
        feat_dim = features.head.fc.in_features
        features.head.fc = nn.Identity()
        for stage in (features.s1, features.s2, features.s3, features.s4):
            for block in stage.children():
                block.conv1 = GatedShift(block.conv1, n_segment=clip_len)
        self._features = features
        self._feat_dim = feat_dim
        self._require_clip_len = clip_len
        hidden_dim = min(feat_dim, MAX_GRU_HIDDEN_DIM)
        self._pred_fine = GRUPrediction(feat_dim, num_classes, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, true_clip_len, channels, height, width = x.shape
        clip_len = true_clip_len
        if true_clip_len < self._require_clip_len:
            x = F.pad(x, (0,) * 7 + (self._require_clip_len - true_clip_len,))
            clip_len = self._require_clip_len
        im_feat = self._features(x.view(-1, channels, height, width)).reshape(batch_size, clip_len, self._feat_dim)
        im_feat = im_feat[:, :true_clip_len]
        return self._pred_fine(im_feat)


def load_model(checkpoint_path: str, config: dict) -> E2ESpot:
    model = E2ESpot(config["num_classes"] + 1, config["feature_arch"], config["clip_len"])
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
