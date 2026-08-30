#!/usr/bin/env python3
"""Build one plot-centered LiDAR tile per field plot for FF3D detection benchmarking.

Extracts a `size`×`size` m square around each of the 13 plot centres, writes .laz into the FF3D
bucket_in_folder (and a scratch backup, since the bucket auto-clears), and records a manifest
(plot_id, centre, field stem count) for the detection parse afterwards.

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=<lazdir> python scripts/ff3d_make_tiles.py [size] [bucket_in_dir]
"""
import sys, shutil
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, laspy, warnings
warnings.simplefilter("ignore")

from greenvista import config
from greenvista.area_based import data as abd

SIZE = float(sys.argv[1]) if len(sys.argv) > 1 else 32.0
BUCKET_IN = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    config.REPO / "project/models/FF3D_inference/FF3D_oracle/bucket_in_folder"
BACKUP = Path(config.LAZ_DIR).parent / "ff3d_tiles"


def main():
    df = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                        config.DATA / "5-dados_campo/inventario_est.csv")
    inv = pd.read_csv(config.DATA / "5-dados_campo/inventario_est.csv")
    df = df.merge(inv[["talhao", "parcela", "n_arv", "n_arv_ha"]], on=["talhao", "parcela"], how="left")
    parc = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        parc[c] = pd.to_numeric(parc[c], errors="coerce").astype("Int64")
    parc["cx"], parc["cy"] = parc.geometry.x, parc.geometry.y
    df = df.merge(parc[["talhao", "parcela", "cx", "cy"]], on=["talhao", "parcela"], how="left")

    BUCKET_IN.mkdir(parents=True, exist_ok=True)
    BACKUP.mkdir(parents=True, exist_ok=True)
    h = SIZE / 2.0
    cloud_hdr, cloud_pts, rows = {}, {}, []
    for _, r in df.sort_values("talhao").iterrows():
        t = int(r.talhao)
        if t not in cloud_pts:
            cloud_pts.clear(); cloud_hdr.clear()
            las = laspy.read(str(config.LAZ_DIR / f"SaoManuelTotal_{t:03d}.laz"))
            cloud_pts[t] = (np.asarray(las.x), np.asarray(las.y)); cloud_hdr[t] = las
        x, y = cloud_pts[t]
        m = (np.abs(x - r.cx) <= h) & (np.abs(y - r.cy) <= h)
        src = cloud_hdr[t]
        sub = laspy.LasData(src.header)
        sub.points = src.points[m].copy()
        name = f"t{t:03d}_p{int(r.parcela):03d}"
        sub.write(str(BUCKET_IN / f"{name}.laz"))
        shutil.copy(BUCKET_IN / f"{name}.laz", BACKUP / f"{name}.laz")
        sz = (BUCKET_IN / f"{name}.laz").stat().st_size / 1e6
        rows.append({"tile": name, "plot_id": r.plot_id, "talhao": t, "parcela": int(r.parcela),
                     "cx": float(r.cx), "cy": float(r.cy), "n_pts": int(m.sum()),
                     "field_n_arv": float(r.n_arv), "field_n_arv_ha": float(r.n_arv_ha),
                     "laz_MB": round(sz, 1)})
        print(f"  {name}: {int(m.sum()):>7} pts, {sz:.1f} MB  (field {int(r.n_arv)} stems)")
    man = pd.DataFrame(rows)
    man.to_csv(BACKUP / "tiles_manifest.csv", index=False)
    print(f"\n{len(man)} tiles → {BUCKET_IN}\n backup + manifest → {BACKUP}")
    print(f" size range {man.laz_MB.min()}–{man.laz_MB.max()} MB, pts {man.n_pts.min()}–{man.n_pts.max()}")


if __name__ == "__main__":
    main()
