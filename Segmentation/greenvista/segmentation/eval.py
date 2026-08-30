"""Score detections / assign known trees to segmented instances."""
import numpy as np

def assign_trees_to_instances(xy, z, inst, tree_xy, radius=1.2, z_lo=2.0, z_hi=12.0, min_pts=5):
    """Assign each known tree (XY) to the FF3D instance occupying its stem column.

    Votes among instances of points within `radius` of the tree and inside a low height band
    [z_lo, z_hi] (the trunk/lower-crown zone, below where an overtopping neighbour's crown dominates),
    which avoids the nearest-centroid failure on suppressed trees. Falls back to all heights if the
    band is sparse. `xy`,`z`,`inst` must share the tree_xy coordinate frame. Returns the instance id
    per tree (-1 when none).
    """
    xy = np.asarray(xy, float); z = np.asarray(z, float); inst = np.asarray(inst)
    out = []
    for e, n in np.asarray(tree_xy, float):
        near = (xy[:, 0] - e) ** 2 + (xy[:, 1] - n) ** 2 <= radius * radius
        band = near & (z >= z_lo) & (z <= z_hi)
        sel = band if band.sum() >= min_pts else near
        ids = inst[sel]; ids = ids[ids >= 0]
        if len(ids) == 0:
            out.append(-1); continue
        vals, cnts = np.unique(ids, return_counts=True)
        out.append(int(vals[np.argmax(cnts)]))
    return np.array(out)

def match_detections(det_xy: np.ndarray, gt_xy: np.ndarray, radius: float = 1.5) -> dict:
    """For each ground-truth position, is there a detection within `radius`?

    det_xy: (M,>=2), gt_xy: (G,2). Returns detection stats + a per-GT boolean `hits` and matched indices.
    """
    det_xy = np.asarray(det_xy, float)
    gt_xy = np.asarray(gt_xy, float)
    hits, idx = [], []
    for g in gt_xy:
        if len(det_xy) == 0:
            hits.append(False); idx.append(-1); continue
        d = np.linalg.norm(det_xy[:, :2] - g, axis=1)
        j = int(np.argmin(d))
        near = d[j] <= radius
        hits.append(bool(near)); idx.append(j if near else -1)
    hits = np.array(hits)
    return {
        "n_gt": int(len(gt_xy)),
        "n_det": int(len(det_xy)),
        "detected": int(hits.sum()),
        "detection_rate": float(hits.mean()) if len(gt_xy) else 0.0,
        "hits": hits,
        "matched_idx": np.array(idx),
    }
