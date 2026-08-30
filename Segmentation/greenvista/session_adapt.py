"""Session-aware, deployment-ordered test-time adaptation of the FF3D post-hoc NMS
merge radius — per stand, unsupervised.

Overlapping tiles + NMS lift FF3D stem detection from 61% to 74% at a fixed global
merge radius r=1.5 m, but that single constant is a compromise: at the tighter
r=1.2 m some stands over-count (duplicates from overlapping tiles survive NMS — e.g.
plots 4_4, 7_14 exceed 110 % of field) while the hardest stands still under-count. No
single global radius is right for every stand, because the right radius depends on the
stand's tree spacing: it must sit above the duplicate scatter (so the same tree seen in
several overlapping tiles collapses to one) and below the true neighbour distance (so
two distinct trees are never merged).

This module adapts that radius online, within a per-stand "session", from the stand's
own detected-centroid statistics — no ground truth, no future tiles:

    radius(stand) = clip( fraction * spacing_estimate(stand), lo, hi )

where `spacing_estimate` is the stand's typical nearest-neighbour spacing, estimated
from the within-tile per-instance NN distances. Within-tile matters: a single FF3D tile
emits about one instance per tree, so its internal NN spacing ≈ the true tree spacing
and is not contaminated by the cross-tile duplicates, which only appear once tiles are
pooled. Denser stands → tighter spacing → tighter radius (protects recall on close
neighbours); sparser stands → looser spacing → looser radius (removes the far-apart
edge-crop duplicates a tight global radius leaves behind).

Two variants are exposed:
  * batch     (`merge_stand_batch`)     — estimate spacing from all of a stand's tiles,
    one radius, pool + NMS. The per-stand summary result.
  * streaming (`merge_stand_streaming`) — tiles arrive in deployment order; at tile k
    the merge radius is derived from spacing over tiles 1..k only, and that tile's
    centroids are merged into a running kept-set. Causal (non-anticipating) by
    construction, which is the deployment form.

Adaptation acts on the post-hoc merge radius rather than on the model because loosening
FF3D's own `sp_score_thr`/`npoint_thr` makes detection worse (masks merge); acting on
the geometry after inference also needs no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "nn_distances",
    "typical_spacing",
    "adaptive_radius",
    "nms_merge",
    "SessionSpacingEstimator",
    "merge_stand_batch",
    "merge_stand_streaming",
    "merge_stand_global",
    "count_within",
]


# --- geometry primitives ---------------------------------------------------------------
def nn_distances(cents) -> np.ndarray:
    """Per-point nearest-neighbour distance within one set of 2-D centroids.

    Returns an empty array if fewer than 2 points (NN undefined)."""
    cents = np.asarray(cents, float).reshape(-1, 2)
    if len(cents) < 2:
        return np.empty((0,))
    d = np.hypot(cents[:, None, 0] - cents[None, :, 0],
                 cents[:, None, 1] - cents[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def typical_spacing(cents, q: float = 50.0) -> float:
    """Robust typical NN spacing of a centroid set = the `q`-th percentile of NN
    distances (median by default). NaN if undefined (<2 points)."""
    nn = nn_distances(cents)
    return float(np.percentile(nn, q)) if len(nn) else float("nan")


def adaptive_radius(spacing: float, fraction: float = 0.80,
                    lo: float = 1.0, hi: float = 2.0, default: float = 1.5) -> float:
    """Merge radius = clip(`fraction` * `spacing`, `lo`, `hi`).

    `fraction` < 1 keeps the radius safely below the neighbour distance; the clamp bounds
    it to the eucalyptus regime (planting ~1.8 m NN → never merge distinct trees, never
    exceed ~one spacing). Falls back to `default` when `spacing` is not finite (e.g. a
    warm-up tile with <2 instances)."""
    if not np.isfinite(spacing):
        return float(default)
    return float(np.clip(fraction * spacing, lo, hi))


def nms_merge(cents, sizes=None, radius: float = 1.5):
    """Greedy NMS: keep the largest instance in each cluster, suppress the rest within
    `radius`. Same rule as the overlapping-tile merge. Returns (kept_xy Kx2, kept_idx K).

    Uses a preallocated kept-array with a vectorised inner distance test — same greedy
    result as the list form, fast enough for the multi-seed synthetic sweep."""
    cents = np.asarray(cents, float).reshape(-1, 2)
    n = len(cents)
    if n == 0:
        return cents, np.empty((0,), int)
    sizes = np.ones(n) if sizes is None else np.asarray(sizes, float)
    order = np.argsort(-sizes)  # biggest crown first — keep the most complete detection
    kept = np.empty((n, 2))
    idx = np.empty(n, int)
    k = 0
    r2 = radius * radius
    for i in order:
        px, py = cents[i, 0], cents[i, 1]
        if k == 0 or np.min((kept[:k, 0] - px) ** 2 + (kept[:k, 1] - py) ** 2) > r2:
            kept[k, 0] = px
            kept[k, 1] = py
            idx[k] = i
            k += 1
    return kept[:k], idx[:k]


def count_within(cents, center, radius: float) -> int:
    """Count centroids within `radius` of `center` (e.g. the 12 m plot count)."""
    cents = np.asarray(cents, float).reshape(-1, 2)
    if len(cents) == 0:
        return 0
    cx, cy = float(center[0]), float(center[1])
    return int((((cents[:, 0] - cx) ** 2 + (cents[:, 1] - cy) ** 2) <= radius * radius).sum())


# --- causal per-stand spacing estimator ------------------------------------------------
@dataclass
class SessionSpacingEstimator:
    """Causal, cumulative estimate of one stand's typical NN spacing from streamed tiles.

    Each tile contributes its within-tile per-instance NN distances (uncontaminated by
    cross-tile duplicates). The estimate is the `q`-th percentile of all within-tile NN
    distances seen so far — a function of past and current tiles only, never future
    tiles, never ground truth. A new stand means a new estimator (a fresh session)."""

    q: float = 50.0
    _pool: list = field(default_factory=list)
    n_tiles: int = 0
    n_pairs: int = 0

    def update(self, tile_cents) -> float:
        """Fold in one tile's centroids; return the current cumulative spacing estimate."""
        d = nn_distances(tile_cents)
        if len(d):
            self._pool.append(d)
            self.n_pairs += len(d)
        self.n_tiles += 1
        return self.spacing()

    def spacing(self) -> float:
        if not self._pool:
            return float("nan")
        return float(np.percentile(np.concatenate(self._pool), self.q))


# --- helpers ---------------------------------------------------------------------------
def _iter_tiles(tile_cents: Sequence, tile_sizes):
    """Yield (cents Nx2, sizes N) per tile; default sizes to ones."""
    for k, tc in enumerate(tile_cents):
        tc = np.asarray(tc, float).reshape(-1, 2)
        if tile_sizes is None:
            ts = np.ones(len(tc))
        else:
            ts = np.asarray(tile_sizes[k], float).reshape(-1)
        yield tc, ts


def _pool(tile_cents: Sequence, tile_sizes):
    cs, ss = [], []
    for tc, ts in _iter_tiles(tile_cents, tile_sizes):
        if len(tc):
            cs.append(tc)
            ss.append(ts)
    if cs:
        return np.vstack(cs), np.concatenate(ss)
    return np.empty((0, 2)), np.empty((0,))


# --- the two adaptive variants + the global baseline -----------------------------------
def merge_stand_batch(tile_cents: Sequence, tile_sizes=None, *, spacing_cents=None,
                      fraction: float = 0.7, lo: float = 1.0, hi: float = 2.0,
                      q: float = 50.0) -> dict:
    """Per-stand adaptive merge (batch): spacing from each tile's within-tile NN → one
    radius → pool + NMS. Unsupervised — sees only detected centroids.

    `spacing_cents` (optional, aligned with `tile_cents`): a cleaner per-tile centroid
    set to estimate spacing from, typically each tile's interior detections, which are
    one per tree and free of the edge-clipped centroid shift that biases the raw
    within-tile NN downward. Merging still uses the full `tile_cents`. Defaults to
    `tile_cents` (raw) when not supplied.

    Returns dict(merged Kx2, radius, spacing, n_pooled)."""
    est = SessionSpacingEstimator(q=q)
    src = tile_cents if spacing_cents is None else spacing_cents
    for tc in src:
        est.update(tc)
    spacing = est.spacing()
    radius = adaptive_radius(spacing, fraction, lo, hi)
    allc, alls = _pool(tile_cents, tile_sizes)
    merged, _ = nms_merge(allc, alls, radius)
    return {"merged": merged, "radius": radius, "spacing": spacing, "n_pooled": len(allc)}


def merge_stand_streaming(tile_cents: Sequence, tile_sizes=None, *, spacing_cents=None,
                          fraction: float = 0.7, lo: float = 1.0, hi: float = 2.0,
                          q: float = 50.0, warmup_default: float = 1.5) -> dict:
    """Per-stand adaptive merge (streaming / causal): tiles in deployment order. At tile
    k the radius comes from spacing over tiles 1..k only (`warmup_default` before any
    pair is seen); that tile's centroids are then merged into the running kept-set. The
    radius at step k never depends on tiles > k or on ground truth.

    `spacing_cents` (optional, aligned with `tile_cents`): cleaner per-tile centroids for
    the spacing estimate (e.g. interior detections); merging still uses `tile_cents`.

    Returns dict(merged Kx2, radii[·per tile], spacings[·per tile], n_pooled)."""
    est = SessionSpacingEstimator(q=q)
    total = sum(len(np.asarray(t).reshape(-1, 2)) for t in tile_cents)
    kept = np.empty((max(total, 1), 2))
    k = 0
    radii, spacings = [], []
    n_pooled = 0
    for idx, (tc, ts) in enumerate(_iter_tiles(tile_cents, tile_sizes)):
        sc = tc if spacing_cents is None else np.asarray(spacing_cents[idx], float).reshape(-1, 2)
        est.update(sc)                                   # causal: past + current only
        spacing = est.spacing()
        radius = adaptive_radius(spacing, fraction, lo, hi, default=warmup_default)
        radii.append(radius)
        spacings.append(spacing)
        n_pooled += len(tc)
        r2 = radius * radius
        for i in np.argsort(-ts):                        # biggest crown first
            px, py = tc[i, 0], tc[i, 1]
            if k == 0 or np.min((kept[:k, 0] - px) ** 2 + (kept[:k, 1] - py) ** 2) > r2:
                kept[k, 0] = px
                kept[k, 1] = py
                k += 1
    return {"merged": kept[:k], "radii": radii, "spacings": spacings, "n_pooled": n_pooled}


def merge_stand_global(tile_cents: Sequence, tile_sizes=None, *, radius: float = 1.5) -> dict:
    """Fixed-global-radius baseline: pool all tiles + NMS at one constant `radius`
    (the 6b behaviour). Returns dict(merged Kx2, radius, n_pooled)."""
    allc, alls = _pool(tile_cents, tile_sizes)
    merged, _ = nms_merge(allc, alls, radius)
    return {"merged": merged, "radius": radius, "n_pooled": len(allc)}
