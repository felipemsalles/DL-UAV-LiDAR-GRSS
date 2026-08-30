"""Read a ForestFormer3D panoptic PLY (instance_pred + semantic_pred) into per-tree point clouds.

Mirrors the v1 loader in project/scripts/tree_analysis.py:load_ply. Running FF3D itself is done via the
Docker container (see project/models/ff3d_blackwell/); this only parses its output.
"""
import numpy as np
from greenvista import config

def load_panoptic_ply(path) -> dict:
    """Return {instance_id: {"points": (N,3), "semantic": (N,)}} for each segmented tree (id >= 0)."""
    from plyfile import PlyData
    ply = PlyData.read(str(path))
    el = ply.elements[0]
    data = {p.name: np.array(el[p.name]) for p in el.properties}
    xyz = np.column_stack([data["x"], data["y"], data["z"]])
    inst = data["instance_pred"]
    sem = data.get("semantic_pred", np.full(len(xyz), config.SEM_LEAF))
    trees = {}
    for iid in np.unique(inst):
        if iid < 0:
            continue
        m = inst == iid
        trees[int(iid)] = {"points": xyz[m], "semantic": sem[m]}
    return trees
