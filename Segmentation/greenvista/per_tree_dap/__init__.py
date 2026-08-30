"""Per-tree DAP (DBH) from UAV-LiDAR crowns — reproduction of a peer's CNN/PointNet/PointTransformer
study on the São Manuel data, evaluated by LOOCV with a shuffle control against an allometry
baseline. N=23 geolocated cubed trees (stand 001). See train.evaluate_all."""
from .data import load_tree_crowns, normalize_points, crown_scalars
from .models import CrownCNN, PointNetReg, PointTransformerReg, DapAllometry, DEEP_MODELS

__all__ = ["load_tree_crowns", "normalize_points", "crown_scalars",
           "CrownCNN", "PointNetReg", "PointTransformerReg", "DapAllometry", "DEEP_MODELS"]
