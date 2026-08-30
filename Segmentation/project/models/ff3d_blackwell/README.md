# ForestFormer3D on Blackwell

This project's contribution to running ForestFormer3D (FF3D) as the tree segmenter.
The FF3D model, its weights and its inference package are not redistributed here.

## Upstream

| | |
|---|---|
| Paper | Xiang et al., ICCV 2025, https://arxiv.org/abs/2506.16991 |
| Code | https://github.com/SmartForest-no/ForestFormer3D |
| Weights | https://zenodo.org/records/16742708 |

FF3D is released under CC BY-NC 4.0, inherited from OneFormer3D. The same license
covers this directory, and it forbids commercial use.

The inference package (`ff3d_forestsens`) is not the public repository and is supplied
by the FF3D authors on request. The changes listed below apply to it.

## Files

**`Dockerfile.blackwell`** builds the container for SM 12.0, on CUDA 12.8 and PyTorch
2.7.1. `cumm` (SM 12.0 patch), `spconv`, `mmcv` and MinkowskiEngine
(`AzharSindhi/MinkowskiEngineCuda13`) are built from source. Build takes 60 to 90 min.

**`tta_adapt.py`** is unsupervised test-time adaptation over the BatchNorm layers of
the backbone, selected by `FF3D_TTA_ADAPT` and inert when unset:

| Mode | Behaviour |
|---|---|
| `adabn` | each tile normalised by its own statistics |
| `session_ema` | running statistics accumulated across tiles, carried on disk |
| `tent` | TENT (Wang et al., 2021) on the BatchNorm affine parameters |

## Changes to the inference package

| File | Change |
|---|---|
| `entrypoint_ff3d.sh` | `weights_only=False` on the `mmengine` 0.7.3 `torch.load` calls, required from PyTorch 2.6 |
| | `torch-cluster` from the PyG index matching the installed torch and CUDA |
| | `--no-build-isolation` for `torch-points-kernels` |
| | site-packages lookup accepts `dist-packages` |
| `oneformer3d.py` | inference `chunk` 1500, to fit 8 GB of VRAM |
| | three guarded call sites for `tta_adapt.py` |
| `run_oracle_pipeline.sh` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| | point coordinates stay local; the float32 buffer cannot hold UTM at 0.2 m |
| `run_docker_locally.sh` | bucket paths self-locate |
| `configs/oneformer3d_qs_radius16_qp300_2many.py` | radius 16 m, 300 query points, chunk 2000 |

## Interface

Clouds go into `FF3D_oracle/bucket_in_folder`. Panoptic maps come back in
`FF3D_oracle/bucket_out_folder` with `x, y, z, instance_pred, semantic_pred`, semantic
classes 0 ground, 1 wood, 2 leaf. `greenvista/segmentation/ff3d.py` parses them.

The output bucket is owned by root and is cleared at the start of every run, so results
must be copied out between runs.
