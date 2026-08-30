"""Tree-by-tree matched metric on stand 001, against the TLS stem map.

A contiguous stem map makes it possible to separate omission from commission, and
not merely compare counts.

Reference: `localizacao_arv.shp`, 892 stems surveyed by terrestrial LiDAR in stand
001, sent by Cristiano on 2026-08-14. Provenance confirmed on 2026-08-17: the
shapefile came from the TLS, and the point cloud went through manual editing after
automatic segmentation.

The manual editing was done on the terrestrial laser cloud, not on the drone cloud,
so the reference is independent of the data being evaluated. Since the operator sees
stems the drone cannot reach, the test becomes harder rather than easier.

Evidence that the reference is consistent: the points fall on trunks in the drone
cloud (5.1x more returns between 1 and 2 m than at random points), the density
matches the census (1,381 against 1,379 trees/ha), the count closes on plot 001
(63 against 63), and it agrees with the 26 GNSS RTX trees to within 16 cm.

Matching: optimal one-to-one assignment (Hungarian), not greedy nearest neighbour,
which overstates hits by allowing two predictions on the same tree.

The predicted position is the instance centroid in both methods, because that is
what FF3D provides. A crown centroid sits metres away from the stem due to lean and
asymmetry, so the 1 m threshold usual in inventory (Liang 2018) is too strict here;
the script reports a sweep of thresholds.

Run:
    PYTHONPATH=. python scripts/exp_matched_metric_talhao001.py
Output: manual_match/matched_metric_talhao001.csv
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import laspy
from scipy.optimize import linear_sum_assignment

from greenvista import config
from scripts.ff3d_detection_overlap import nms_merge

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
SAT_OUT = config.REPO / "project/models/SegmentAnyTree_blackwell/bucket_out"
CENTROIDS = config.REPO / "data/detections/ff3d_overlap_centroids_13plot.csv"
RAIO = float(np.sqrt(400 / np.pi))          # 11.28 m, the 400 m² plot
TILE = 32.0                                 # the tile both methods received
MARGEM = 3.0                                # inner buffer against edge effects
LIMIARES = [1.0, 1.5, 2.0, 2.5, 3.0]
PLOTS = ["1_1", "1_2"]


def centros():
    p = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    p["t"] = pd.to_numeric(p.talhao, errors="coerce")
    p["pa"] = pd.to_numeric(p.parcela, errors="coerce")
    return {f"{int(r.t)}_{int(r.pa)}": (r.geometry.x, r.geometry.y)
            for _, r in p.dropna(subset=["t", "pa"]).iterrows()}


def no_circulo(xy, cx, cy, raio=RAIO):
    return xy[np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) <= raio]


def na_area(xy, cx, cy, meia=TILE / 2 - MARGEM):
    """Central square of the tile, inset by MARGEM on each side.

    Evaluation is done here, and not on the 400 m² circle, for two reasons. The
    inset removes the edge effect without needing a special margin rule (without it
    precision can even exceed 1), and the usable area rises from 400 to 676 m²,
    which nearly doubles the number of trees evaluated. The rule is identical for
    both methods and for the reference.
    """
    if len(xy) == 0:
        return xy
    return xy[(np.abs(xy[:, 0] - cx) <= meia) & (np.abs(xy[:, 1] - cy) <= meia)]


def posicoes_sat(pid):
    """Centroid of each instance predicted by SegmentAnyTree."""
    f = SAT_OUT / f"plot_{pid}_out.laz"
    las = laspy.read(str(f))
    inst = np.asarray(las.PredInstance)
    xy = np.column_stack([np.asarray(las.x), np.asarray(las.y)])
    return np.array([xy[inst == i].mean(0) for i in np.unique(inst[inst > 0])])


def posicoes_ff3d(pid, raio_fusao=1.5):
    """Centroid of each instance of our method, after fusing the 9 views."""
    c = pd.read_csv(CENTROIDS)
    g = c[c.plot_id == pid]
    return nms_merge(g[["cx", "cy"]].to_numpy(float), g["n_pts"].to_numpy(float), raio_fusao)


def parear(ref, pred, limiar):
    """Optimal one-to-one assignment. Returns (n_matched, positional RMSE)."""
    if len(ref) == 0 or len(pred) == 0:
        return 0, float("nan")
    d = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    custo = np.where(d <= limiar, d, 1e6)
    li, ci = linear_sum_assignment(custo)
    ok = d[li, ci] <= limiar
    return int(ok.sum()), float(np.sqrt((d[li, ci][ok] ** 2).mean())) if ok.any() else float("nan")


def main():
    ref_todo = gpd.read_file(MAPA)
    REF = np.column_stack([ref_todo.geometry.x, ref_todo.geometry.y])
    ctr = centros()

    linhas = []
    for pid in PLOTS:
        cx, cy = ctr[pid]
        ref = na_area(REF, cx, cy)
        for nome, pred_todo in [("SegmentAnyTree", posicoes_sat(pid)),
                                ("Nosso metodo (FF3D+MVA)", posicoes_ff3d(pid))]:
            pred = na_area(pred_todo, cx, cy)
            for lim in LIMIARES:
                tp, rmse = parear(ref, pred, lim)
                fn, fp = len(ref) - tp, len(pred) - tp
                prec = tp / len(pred) if len(pred) else 0.0
                rec = tp / len(ref) if len(ref) else 0.0
                f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
                linhas.append(dict(parcela=pid, metodo=nome, limiar_m=lim,
                                   ref=len(ref), pred=len(pred), TP=tp, FP=fp, FN=fn,
                                   precisao=prec, revocacao=rec, F1=f1, rmse_pos_m=rmse))
    df = pd.DataFrame(linhas)

    print(f"reference: stem map, {len(REF)} trees in stand 001")
    print(f"evaluation area: central square of {TILE-2*MARGEM:.0f} m, {(TILE-2*MARGEM)**2:.0f} m²\n")
    for pid in PLOTS:
        sub = df[df.parcela == pid]
        print(f"--- plot {pid}: {sub.ref.iloc[0]} stems in the reference")
        print(f"{'method':>24} {'thr':>5} {'pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} "
              f"{'prec':>6} {'recall':>6} {'F1':>6} {'RMSE':>6}")
        for _, r in sub.iterrows():
            print(f"{r.metodo:>24} {r.limiar_m:>5.1f} {r.pred:>5} {r.TP:>4} {r.FP:>4} {r.FN:>4} "
                  f"{r.precisao:>6.3f} {r.revocacao:>6.3f} {r.F1:>6.3f} {r.rmse_pos_m:>6.2f}")
        print()

    tot = (df.groupby(["metodo", "limiar_m"])[["ref", "pred", "TP", "FP", "FN"]].sum().reset_index())
    tot["precisao"] = tot.TP / tot.pred
    tot["revocacao"] = tot.TP / tot.ref
    tot["F1"] = 2 * tot.precisao * tot.revocacao / (tot.precisao + tot.revocacao)
    print("=== both plots pooled ===")
    print(f"{'method':>24} {'thr':>5} {'pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'prec':>6} {'recall':>6} {'F1':>6}")
    for _, r in tot.sort_values(["metodo", "limiar_m"]).iterrows():
        print(f"{r.metodo:>24} {r.limiar_m:>5.1f} {int(r.pred):>5} {int(r.TP):>4} {int(r.FP):>4} "
              f"{int(r.FN):>4} {r.precisao:>6.3f} {r.revocacao:>6.3f} {r.F1:>6.3f}")

    out = config.OUT_DIR / "matched_metric_talhao001.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
