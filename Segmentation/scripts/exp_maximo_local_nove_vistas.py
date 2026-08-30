#!/usr/bin/env python3
"""Classic local maximum with the same nine views and the same NMS fusion as the networks.

Table II of the paper gives the two networks two stages that the classic baseline does
not get: nine-view aggregation with NMS fusion, and test-time adaptation. The adaptation
does not in fact apply, since the classic detector has no normalisation layer to adapt,
but nine-view aggregation is method-agnostic. This script runs the local maximum on the
nine views and fuses it the same way, so as to separate the effect of the two stages from
the effect of the model choice. Reference: the classic scores 0.792 on a single pass and
the networks reach 0.921.

Protocol, identical to that of exp_flight_study_maximo_local.py:
  * source   work/ff3d_tiles_overlap/, the same nine classified tiles the networks saw
             before convert_ff3d_tiles.py stripped the extra fields
  * LM       build_chm + void filling + Gaussian smoothing sigma 0.5
  * yardstick  matched F1, Hungarian assignment at 2 m, central 26 m square of the two
             plots that have a stem map, the same 211 stems as in Table II

NMS weight: the networks fuse by keeping the instance with the largest point count. The
local maximum has no instance, so the analogue is peak height, the only quantity the
detector produces per detection. Keeping the tallest peak applies the same rule as on the
network side: the strongest candidate wins.

The sweep is generous on purpose: fusion radius and window are swept jointly and the best
F1 is reported. The networks only had the radius swept, because their window is
architecture rather than a parameter. This way the answer does not hinge on a badly chosen
window for the new condition.

Run: PYTHONPATH=. python scripts/exp_maximo_local_nove_vistas.py
Output: manual_match/maximo_local_nove_vistas.csv
"""
import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402

# reuse the already validated harness instead of reimplementing it and silently diverging
_s = importlib.util.spec_from_file_location(
    "mlbase", R / "scripts/exp_flight_study_maximo_local.py")
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)
le_vegetacao, topos, na_area, parear = _m.le_vegetacao, _m.topos, _m.na_area, _m.parear

sys.path.insert(0, str(R / "scripts"))
from ff3d_detection_overlap import nms_merge  # noqa: E402

VISTAS = R / "work/ff3d_tiles_overlap"
UMA = R / "work/ff3d_degraded/full"
MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
TILES = ["t001_p001", "t001_p002"]
RESOLUCOES = [0.25, 0.5]
JANELAS = [3, 5, 7, 9]
RAIOS = [round(x, 2) for x in np.arange(0.5, 3.01, 0.1)]
SAIDA = config.OUT_DIR / "maximo_local_nove_vistas.csv"


def pontuar(preds_por_tile, centros, REF):
    """P, R, F1 pooled over the two plots, same rule as the rest of the project."""
    tp = n_ref = n_pred = 0
    for t in TILES:
        cx, cy, _ = centros[t]
        ref = na_area(REF, cx, cy)
        pred = na_area(preds_por_tile[t], cx, cy)
        tp += parear(ref, pred)
        n_ref += len(ref)
        n_pred += len(pred)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_ref if n_ref else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), tp, n_pred, n_ref


def main():
    centros = _m.centros_e_censo()
    ref_todo = gpd.read_file(MAPA)
    REF = np.column_stack([ref_todo.geometry.x, ref_todo.geometry.y])

    print("reading the point clouds", flush=True)
    nove = {t: [le_vegetacao(VISTAS / f"{t}_g{g}.laz") for g in range(9)] for t in TILES}
    uma = {t: le_vegetacao(UMA / f"{t}.laz") for t in TILES}

    linhas = []

    # ---- 1. reproduce the single pass, to prove the harness matches Table II
    print("\n[1] single pass, should reproduce 0.792 at 0.25 m / window 3 (0.75 m)"
          " and 0.691 at window 7 (1.75 m)", flush=True)
    for res in RESOLUCOES:
        for jan in JANELAS:
            pr = {t: topos(*uma[t], res, jan)[:, :2] for t in TILES}
            p, r, f1, tp, npd, nrf = pontuar(pr, centros, REF)
            linhas.append(dict(cond="uma_passada", res_m=res, janela_px=jan,
                               janela_m=jan * res, raio_fusao=np.nan, n_ref=nrf,
                               n_pred=npd, tp=tp, precisao=p, revocacao=r, f1=f1))
            print(f"   res {res} win {jan} ({jan*res:.2f} m): P {p:.3f} R {r:.3f} F1 {f1:.3f}",
                  flush=True)

    # ---- 2. nine views fused by NMS, sweeping window AND radius
    print("\n[2] nine views + NMS, sweeping window and fusion radius", flush=True)
    for res in RESOLUCOES:
        for jan in JANELAS:
            brutos = {}
            for t in TILES:
                acc = [topos(xy, z, res, jan) for xy, z in nove[t]]
                acc = [a for a in acc if len(a)]
                brutos[t] = np.vstack(acc) if acc else np.empty((0, 3))
            melhor = None
            for raio in RAIOS:
                pr = {t: nms_merge(brutos[t][:, :2], brutos[t][:, 2], raio)
                      for t in TILES}
                p, r, f1, tp, npd, nrf = pontuar(pr, centros, REF)
                linhas.append(dict(cond="nove_vistas", res_m=res, janela_px=jan,
                                   janela_m=jan * res, raio_fusao=raio, n_ref=nrf,
                                   n_pred=npd, tp=tp, precisao=p, revocacao=r, f1=f1))
                if melhor is None or f1 > melhor[0]:
                    melhor = (f1, raio, p, r, npd)
            f1, raio, p, r, npd = melhor
            print(f"   res {res} win {jan} ({jan*res:.2f} m): best radius {raio:.1f} m, "
                  f"pred {npd}, P {p:.3f} R {r:.3f} F1 {f1:.3f}", flush=True)

    df = pd.DataFrame(linhas)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAIDA, index=False)

    print("\n" + "=" * 62)
    u = df[df.cond == "uma_passada"].sort_values("f1", ascending=False).iloc[0]
    n = df[df.cond == "nove_vistas"].sort_values("f1", ascending=False).iloc[0]
    print(f"BEST single pass : F1 {u.f1:.3f}  (res {u.res_m}, window {u.janela_m:.2f} m)")
    print(f"BEST nine views  : F1 {n.f1:.3f}  (res {n.res_m}, window {n.janela_m:.2f} m, "
          f"radius {n.raio_fusao:.1f} m)")
    print(f"GAIN from nine views: {(n.f1 - u.f1) * 100:+.1f} F1 points")
    print(f"compare with SAT 0.811 -> 0.870 (+5.9) and FF3D 0.681 -> 0.828 (+14.7)")
    print(f"wrote {SAIDA}")


if __name__ == "__main__":
    main()
