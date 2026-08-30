"""Compact hybrid multi-task CNN for plot volume.

Small on purpose, since overfitting is the risk at N=13: 4 conv blocks → global average pool, then
the pooled features are concatenated with the absolute-height scalar side-features
(zmax/zmean/zsd) before two regression heads — `Vol_ha` and `D_cm` (DAP). The scalar branch keeps the
strong linear height signal always available, so the conv stack only has to add canopy-texture
information on top of that floor.

Inputs are expected pre-standardized (train.py fits the stats on the training fold only); the heads
predict in standardized target space.
"""
import torch
import torch.nn as nn


def _block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2)
    )


class PlotCNN(nn.Module):
    def __init__(self, in_ch=3, n_scalars=3, n_targets=2, base=16, p_drop=0.3):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_ch, base),        # 128 -> 64
            _block(base, base * 2),     # 64 -> 32
            _block(base * 2, base * 4), # 32 -> 16
            _block(base * 4, base * 4), # 16 -> 8
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(base * 4 + n_scalars, 64), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(64, n_targets),
        )

    def forward(self, raster, scalars):
        z = self.gap(self.features(raster)).flatten(1)   # (B, base*4)
        z = torch.cat([z, scalars], dim=1)               # hybrid concat
        return self.head(z)                              # (B, n_targets) = [Vol_ha, D_cm], std space
