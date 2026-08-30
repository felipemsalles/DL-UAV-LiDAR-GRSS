#!/usr/bin/env python3
"""Exports the detected trees as a shapefile, for third-party checking.

It goes out with no matching field at all: the recipient redoes the matching against their own map,
and shipping the "casou" column alongside would hand over the answer, removing the independence of
the check. Only attributes the drone cloud produces on its own are included.

The .prj is written explicitly. The stem map we received declares no projection, which forces
whoever opens it to guess the system; the coordinates agree with SIRGAS 2000 / UTM 22S to the
millimetre. A GeoPackage is written as well, which carries the projection without a sidecar file.

Stand 001 is written to a separate file in addition to the full set, because it is the only one with
a field map on the other side and it is where the comparison closes.

The zip carries a read-me. A shapefile stores geometry and attributes, and nothing about who
produced it; without the accompanying text there is no way to say which model generated the file,
whether any filter was applied, or which flight it came from.

Run: PYTHONPATH=. python scripts/exporta_deteccoes_shp.py
Out: figs_en/entrega_cristiano/shapefile/ and the zip arvores_detectadas_SaoManuel.zip
"""
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenvista import config  # noqa: E402

DETEC = config.REPO / "data/detections/sat_w2w_arvores.csv"
SAIDA = config.ENTREGA_DIR / "shapefile"
# The zip is generated here, not by hand: a shapefile is six files that only work together, and
# zipping manually has already made the name diverge from the script.
ZIP = config.ENTREGA_DIR / "arvores_detectadas_SaoManuel"
# field names of at most 10 characters, otherwise the shapefile driver truncates them silently
CAMPOS = {"arv_id": "arv_id", "talhao": "talhao", "tile": "tile",
          "altura_m": "altura_m", "n_pontos": "n_pontos", "d_divisa": "d_divisa"}


def leia_me(g) -> str:
    """Text that travels inside the zip. Written for someone who opens it with no context."""
    baixas = int((g.altura_m < 5).sum())
    return f"""AUTOMATICALLY DETECTED TREES, SAO MANUEL FARM

GreenVista project, ICMC/USP.

WHAT THIS IS
{len(g)} trees detected in a drone LiDAR point cloud, covering the six stands
delivered, 9.87 ha in total. Each point is the estimated base of one tree.

HOW IT WAS GENERATED
Model       SegmentAnyTree, instance-segmentation network, weights published by
            the authors, with no training and no fine-tuning on these data
Adaptation  AdaBN, the model recomputes the normalisation statistics on the
            cloud itself before each inference
Aggregation each tree is seen 9 times, in 32 m tiles at a stride of 10.67 m
Merging     non-maximum suppression with a radius of 1.7 m
Cloud       DJI Zenmuse L2, flight of 2025-02-08, 418 to 1112 points per m2

NO FILTER WAS APPLIED
Every instance the model produced is included. {baixas} of them are less than 5 m
tall and are understorey fragments, not trees. The fields altura_m and n_pontos
allow filtering according to use.

FIELDS
arv_id     identifier, in the format T<stand>_<sequence>
talhao     stand number
tile       cloud tile in which the instance was detected
altura_m   maximum height of the instance, in metres above ground
n_pontos   number of cloud points assigned to the instance
d_divisa   distance to the stand boundary, in metres

PROJECTION
SIRGAS 2000 / UTM zone 22S, EPSG:31982, declared in the .prj and in the GeoPackage.

FILES
arvores_detectadas_todos_talhoes.shp/.gpkg   the six stands
arvores_detectadas_talhao001.shp/.gpkg       stand 001 only

KNOWN ACCURACY
In stand 001, compared with the terrestrial laser survey of 892 trees, the method
found 864 of them. The detected position lies 0.60 m from the stem at the median,
because it is estimated from the crown seen from above.
"""


def main():
    d = pd.read_csv(DETEC).reset_index(drop=True)
    g = gpd.GeoDataFrame(
        pd.DataFrame({
            "arv_id": [f"T{t:03d}_{i:05d}" for i, t in enumerate(d.talhao, 1)],
            "talhao": d.talhao.astype(int),
            "tile": d.tile,
            "altura_m": d.z_max.round(2),
            "n_pontos": d.n_pts.astype(int),
            "d_divisa": d.dist_divisa.round(2),
        }),
        geometry=gpd.points_from_xy(d.base_x, d.base_y),
        crs=config.CRS,
    )
    SAIDA.mkdir(parents=True, exist_ok=True)
    for rot, sub in (("todos_talhoes", g), ("talhao001", g[g.talhao == 1])):
        for ext, drv in ((".shp", "ESRI Shapefile"), (".gpkg", "GPKG")):
            sub.to_file(SAIDA / f"arvores_detectadas_{rot}{ext}", driver=drv)
        print(f"{rot}: {len(sub)} trees")
    (SAIDA / "LEIA-ME.txt").write_text(leia_me(g), encoding="utf-8")
    zipado = shutil.make_archive(str(ZIP), "zip", root_dir=SAIDA)
    print("\nper stand:", g.talhao.value_counts().sort_index().to_dict())
    print(SAIDA)
    print(f"{zipado} ({Path(zipado).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
