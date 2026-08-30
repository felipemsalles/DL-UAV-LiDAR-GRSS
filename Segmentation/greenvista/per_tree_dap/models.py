"""Per-tree DAP regressors: the peer's three deep architectures + allometry baselines.

The deep nets are transcribed from the peer's block diagram:
  * CrownCNN         — Conv2D 16→32→64 (3×3, ReLU, MaxPool) → flatten → dense head → DAP.
  * PointNetReg      — shared MLP 3→64→128→256 (Conv1d) → global max pool → 256→128→64→1.
  * PointTransformerReg — point embed → multi-head self-attention (dim 128, 4 heads) → FF (128→256→128)
                          with two residuals → global mean pool → 128→64→1.

At N=23 these are deliberately kept as small as the diagram; the question under test is whether any
of them beats the allometry baseline under LOOCV.
"""
import numpy as np
import torch.nn as nn


# ------------------------------------------------------------------ deep nets ------------------
class CrownCNN(nn.Module):
    """2D raster (height/density/spread) → DAP. Mirrors the peer's CNN column."""
    def __init__(self, in_ch=3, size=32, p_drop=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * (size // 8) ** 2, 64),
                                  nn.ReLU(inplace=True), nn.Dropout(p_drop), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


class PointNetReg(nn.Module):
    """Vanilla PointNet regressor (no T-nets — matches the peer's minimal diagram).

    `n_targets` lets the SSL pretext head (K crown metrics) and the DAP head (1) share the `mlp`
    backbone; `_load_compatible` transfers the backbone by matching names+shapes."""
    def __init__(self, p_drop=0.3, n_targets=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(p_drop), nn.Linear(128, 64), nn.ReLU(inplace=True),
                                  nn.Linear(64, n_targets))

    def forward(self, x):                 # x: (B, N, 3)
        f = self.mlp(x.transpose(1, 2))   # (B, 256, N)
        g = f.max(dim=2).values           # global max pool → (B, 256)
        return self.head(g).squeeze(-1)


class PointTransformerReg(nn.Module):
    """Lightweight point-transformer regressor: embed → MHSA → FF, two residuals → mean pool → head."""
    def __init__(self, dim=128, heads=4, p_drop=0.3):
        super().__init__()
        self.embed = nn.Linear(3, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(inplace=True), nn.Linear(256, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(inplace=True),
                                  nn.Dropout(p_drop), nn.Linear(64, 1))

    def forward(self, x):                 # x: (B, N, 3)
        h = self.embed(x)
        a, _ = self.attn(h, h, h)
        h = self.norm1(h + a)             # residual 1
        h = self.norm2(h + self.ff(h))    # residual 2
        g = h.mean(dim=1)                 # global mean pool
        return self.head(g).squeeze(-1)


DEEP_MODELS = {"CNN": CrownCNN, "PointNet": PointNetReg, "PointTransformer": PointTransformerReg}


# ------------------------------------------------------------------ allometry baselines --------
class DapAllometry:
    """ln(DAP) = a + Σ b_i ln(x_i) on crown scalars (zmax / zmean / crown_area). The floor the deep
    nets have to beat. Operates on a list of crown-scalar dicts."""
    def __init__(self, cols=("zmax", "crown_area")):
        self.cols = list(cols)

    def _design(self, feats):
        L = np.log(np.array([[f[c] for c in self.cols] for f in feats], float))
        return np.column_stack([np.ones(len(L)), L])

    def fit(self, feats, dap):
        A = self._design(feats)
        self.coef_, *_ = np.linalg.lstsq(A, np.log(np.asarray(dap, float)), rcond=None)
        return self

    def predict(self, feats):
        return np.exp(self._design(feats) @ self.coef_)
