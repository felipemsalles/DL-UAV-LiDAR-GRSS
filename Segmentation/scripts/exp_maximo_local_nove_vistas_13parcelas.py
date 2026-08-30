#!/usr/bin/env python3
"""Local maximum with nine views over the 13 plots, for the count table.

In the count table the network rows use nine views plus AdaBN while the local-maximum
rows use a single pass, the same asymmetry already corrected in the matching table
(exp_maximo_local_nove_vistas.py). Here the baseline gets the same treatment.

Protocol, identical to the tiler scripts/ff3d_make_tiles_overlap.py:
  * nine 32 m windows centred on (cx+dx, cy+dy), dx,dy in {-16, 0, +16}
  * square clip, |x-tcx| <= 16 and |y-tcy| <= 16, straight from the stand point cloud
  * ground and noise removed by classes 2 and 18, CHM filled, Gaussian sigma 0.5
  * NMS fusion with peak height as the weight, as in the matching script
  * count inside the 11.28 m circle, the 400 m2 the field crew measured

The fusion radius is fixed at 1.1 m and is not swept here. It was chosen once, by matched
F1 on the stem map of stand 001, in the matching script; sweeping it again on the count
would mean tuning a parameter on the very quantity being reported. The networks follow the
same rule, each at the radius that matched F1 selected.

Validation: the single-pass condition must reproduce the numbers already published in the
table (97.8% / 53.1% / 47.7%) before the new condition means anything. The first two
reproduce to the decimal; the third gives 51.2% against the published 47.7%. The cause is
the Gaussian sigma: the two swept rows use 0.5, the value da Cunha Neto reports, while the
"historical setting" row used 0.6. With sigma 0.6 this script reproduces exactly 47.7% and
MAE 28.8.

The third row therefore differed from the other two in two parameters, only one of them
declared, and sigma never entered the sweep. Harmonised at 0.5 it becomes 61.4% / 21.3,
practically identical to the 1.75 m window row (61.9% / 21.0).

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=/tmp/lazall \
     python scripts/exp_maximo_local_nove_vistas_13parcelas.py
Output: manual_match/maximo_local_nove_vistas_13parcelas.csv
"""
import importlib.util
import itertools
import math
import sys
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402

_s = importlib.util.spec_from_file_location(
    "mlbase", R / "scripts/exp_flight_study_maximo_local.py")
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
topos, centros_e_censo = _m.topos, _m.centros_e_censo

sys.path.insert(0, str(R / "scripts"))
from ff3d_detection_overlap import nms_merge  # noqa: E402

PR = math.sqrt(400 / math.pi)                  # 11.28 m
SIZE, STRIDE = 32.0, 16.0
GRID = list(itertools.product([-STRIDE, 0.0, STRIDE], [-STRIDE, 0.0, STRIDE]))
CENTRO = GRID.index((0.0, 0.0))                # view g4, which is the single pass
RAIO_FUSAO = 1.1                               # chosen by matched F1, not here
AJUSTES = [("0.75 m window", 0.25, 3), ("1.75 m window", 0.25, 7),
           ("1.5 m, coarse raster", 0.5, 3)]
PUBLICADO = {"0.75 m window": 97.8, "1.75 m window": 53.1, "1.5 m, coarse raster": 47.7}
SAIDA = config.OUT_DIR / "maximo_local_nove_vistas_13parcelas.csv"


def conta_no_circulo(xy, cx, cy):
    if len(xy) == 0:
        return 0
    return int((np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) <= PR).sum())


def main():
    centros = centros_e_censo()
    alvos = sorted(k for k in centros if not np.isnan(centros[k][2]))
    print(f"{len(alvos)} plots: {', '.join(alvos)}\n", flush=True)

    # nine clips per plot, one read per stand
    vistas = {}
    for tal in sorted({k[1:4] for k in alvos}):
        f = Path(config.LAZ_DIR) / f"SaoManuelTotal_{tal}.laz"
        print(f"reading {f.name}", flush=True)
        las = laspy.read(str(f))
        cls = np.asarray(las.classification)
        veg = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
        x, y, z = (np.asarray(las.x)[veg], np.asarray(las.y)[veg], np.asarray(las.z)[veg])
        del las, cls, veg
        for pid in [k for k in alvos if k[1:4] == tal]:
            cx, cy, _ = centros[pid]
            jan = []
            for dx, dy in GRID:
                m = (np.abs(x - (cx + dx)) <= SIZE / 2) & (np.abs(y - (cy + dy)) <= SIZE / 2)
                jan.append((np.column_stack([x[m], y[m]]), z[m]))
            vistas[pid] = jan
            print(f"  {pid}: {[len(v[1]) for v in jan]}", flush=True)
        del x, y, z

    linhas = []
    for nome, res, jan_px in AJUSTES:
        for cond in ("uma_passada", "nove_vistas"):
            det, campo = [], []
            for pid in alvos:
                cx, cy, n_campo = centros[pid]
                if cond == "uma_passada":
                    t = topos(*vistas[pid][CENTRO], res, jan_px)
                    xy = t[:, :2] if len(t) else np.empty((0, 2))
                else:
                    acc = [topos(*v, res, jan_px) for v in vistas[pid]]
                    acc = [a for a in acc if len(a)]
                    br = np.vstack(acc) if acc else np.empty((0, 3))
                    xy = nms_merge(br[:, :2], br[:, 2], RAIO_FUSAO)
                n = conta_no_circulo(xy, cx, cy)
                det.append(n); campo.append(float(n_campo))
                linhas.append(dict(ajuste=nome, condicao=cond, parcela=pid,
                                   detectado=n, campo=n_campo))
            det, campo = np.array(det), np.array(campo)
            razao = 100 * det.sum() / campo.sum()
            mae = float(np.abs(det - campo).mean())
            marca = ""
            if cond == "uma_passada":
                d = razao - PUBLICADO[nome]
                marca = f"   [published {PUBLICADO[nome]}%, delta {d:+.1f}]"
            print(f"{nome:24s} {cond:12s} ratio {razao:6.1f}%  MAE {mae:5.1f}{marca}",
                  flush=True)

    df = pd.DataFrame(linhas)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAIDA, index=False)
    g = (df.groupby(["ajuste", "condicao"])
           .apply(lambda d: pd.Series({
               "razao_pct": 100 * d.detectado.sum() / d.campo.sum(),
               "mae_arv": (d.detectado - d.campo).abs().mean()}), include_groups=False))
    print("\n" + g.round(1).to_string())
    print(f"\nwrote {SAIDA}")


if __name__ == "__main__":
    main()
