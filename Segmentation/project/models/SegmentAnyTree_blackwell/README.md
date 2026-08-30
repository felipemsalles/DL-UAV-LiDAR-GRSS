# SegmentAnyTree on Blackwell

This project's contribution to running SegmentAnyTree as the second tree segmenter,
for the network-against-network comparison. The model and its weights are not
redistributed here.

## Upstream

https://github.com/SmartForest-no/SegmentAnyTree, built on torch-points3d
(BSD, Principia Labs Ltd). `fetch_source.sh` clones it into `src/`.

## Files

| File | Purpose |
|---|---|
| `Dockerfile.blackwell` | image for SM 12.0, `FROM ff3d-blackwell` for the compiled MinkowskiEngine, spconv and mmcv; adds the torch-points3d Python layer |
| `fetch_source.sh` | clones the upstream source |
| `convert_ff3d_tiles.py` | rewrites the FF3D tiles into the format SegmentAnyTree reads, so both models see the same data |
| `prepare_input.py` | writes minimal LAS, point format 0, in absolute coordinates |
| `sat_eval.py` | evaluation, with optional AdaBN and TENT behind `SAT_TTA` |
| `sat_pipeline.sh` | inference orchestrator |
| `run_sat_locally.sh` | entry point; `SAT_IN` is the input directory, `SAT_OUT` the output |

## Build

```bash
docker build -f Dockerfile.blackwell -t sat-blackwell .
```

Requires the `ff3d-blackwell` image, built from `../ff3d_blackwell/Dockerfile.blackwell`.
