"""Rasterize→CNN plot-level eucalyptus volume (segmentation-free, no per-tree coords).

Pipeline: per plot, extract the circular footprint from the cloud (footprint.py), render a
multi-channel canopy image keeping absolute height in metres (render.py), then regress Vol_ha
(+ DAP as a multi-task auxiliary) with a compact hybrid CNN. See
the design notes for the design and the evaluation contract.
"""
from .footprint import extract_footprint
from .render import to_raster, N_CHANNELS, CHM, DENSITY, SPREAD

__all__ = ["extract_footprint", "to_raster", "N_CHANNELS", "CHM", "DENSITY", "SPREAD"]
