"""The FF3D noise threshold, swept and validated outside the test set.

`tools/merge_prediction.py:323` fixes `threshold = 200` on `score = volume / h_min`,
where `h_min` is the height above ground of the lowest point of the instance. The intent
is to discard floating crown blobs: a real tree touches the ground, `h_min` is almost
zero and the score blows up.

But a suppressed tree in an aerial scan has no trunk returned by the laser: its lowest
point sits 10 or 15 metres up, the score collapses and the filter removes it. That is,
the filter discards exactly the trees this work is looking for.

One reading with 3 of the 9 views gave F1 78.1% at threshold 200 against 93.9% at
threshold 25, with precision rising to 100%. Here the test is redone with all 9 views and
with the threshold selected on one plot and measured on the other.

Two validation precautions:
  1. the threshold is selected on 1_1 and measured on 1_2, and vice versa, never in the
     same place where it is reported;
  2. the fusion radius is held at the value FF3D has already been shown to prefer, so as
     not to mix two effects and credit the threshold with a gain that belongs to the radius.

Input: `manual_match/ff3d_noisyscore/*_noisysegments.ply`, saved during the run by the
collector, because the pipeline deletes that directory at the end.

Run: PYTHONPATH=. python scripts/exp_limiar_ruido_ff3d.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_tta_comparison import MAPA, PLOTS, centros_e_censo, no_quadrado, parear  # noqa: E402
from ff3d_detection_overlap import nms_merge  # noqa: E402

from greenvista import config  # noqa: E402

ESCORES = config.OUT_DIR / "ff3d_noisyscore"
BACKUP = Path(config.LAZ_DIR).parent / "ff3d_tiles_overlap"
LIMIARES = [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 800]
RAIO_FUSAO = 1.1     # the value FF3D preferred in the radius validation; fixed on purpose
LIMIAR_CASADO = 2.0


def carrega(pid):
    """Instances from every view of one plot, with score and UTM position."""
    t, p = pid.split("_")
    padrao = f"t{int(t):03d}_p{int(p):03d}_g*_noisysegments.ply"
    saida = []
    for f in sorted(ESCORES.glob(padrao)):
        tile = f.name.split("_noisysegments")[0]
        laz = BACKUP / f"{tile}.laz"
        if not laz.exists():
            print(f"  no backup tile for {tile}, skipped")
            continue
        el = PlyData.read(str(f)).elements[0]
        d = {q.name: np.asarray(el[q.name]) for q in el.properties}
        las = laspy.read(str(laz))
        mx, my = float(np.asarray(las.x).mean()), float(np.asarray(las.y).mean())
        inst = d.get("instance_pred", d.get("instance"))
        sc = d["instance_score"]
        for i in np.unique(inst[inst >= 0]):
            m = inst == i
            saida.append((float(d["x"][m].mean() + mx), float(d["y"][m].mean() + my),
                          int(m.sum()), float(np.nanmedian(sc[m]))))
    return pd.DataFrame(saida, columns=["cx", "cy", "n_pts", "escore"])


def mede(inst, pid, limiar, ref, ctr):
    s = inst[inst.escore >= limiar]
    cx, cy, _ = ctr[pid]
    r = no_quadrado(ref, cx, cy)
    if len(s) == 0:
        return 0.0, 0.0, 0.0, 0, len(r)
    pred = no_quadrado(nms_merge(s[["cx", "cy"]].to_numpy(float),
                                 s.n_pts.to_numpy(float), RAIO_FUSAO), cx, cy)
    tp, _ = parear(r, pred, LIMIAR_CASADO)
    p = tp / len(pred) if len(pred) else 0.0
    rc = tp / len(r) if len(r) else 0.0
    return (2 * p * rc / (p + rc) if p + rc else 0.0), p, rc, len(pred), len(r)


def main():
    ctr = centros_e_censo()
    ref_all = gpd.read_file(MAPA)
    REF = np.column_stack([ref_all.geometry.x, ref_all.geometry.y])

    inst = {}
    for pid in PLOTS:
        inst[pid] = carrega(pid)
        n_vistas = len(list(ESCORES.glob(
            f"t{int(pid.split('_')[0]):03d}_p{int(pid.split('_')[1]):03d}_g*_noisysegments.ply")))
        print(f"plot {pid}: {n_vistas} views, {len(inst[pid])} instances before the cut")
    if any(len(v) == 0 for v in inst.values()):
        raise SystemExit("scores are missing for some plot; the run did not finish")

    print(f"\nfusion radius FIXED at {RAIO_FUSAO} m, matching threshold {LIMIAR_CASADO} m\n")
    print("=== full sweep, per plot (in-sample, only to see the shape) ===")
    linhas = []
    for pid in PLOTS:
        print(f"\n-- plot {pid}")
        print(f"{'thresh':>7} {'inst':>6} {'pred':>6} {'prec':>7} {'recall':>7} {'F1':>7}")
        for lim in LIMIARES:
            f1, p, r, npred, nref = mede(inst[pid], pid, lim, REF, ctr)
            linhas.append(dict(parcela=pid, limiar=lim, F1=f1, precisao=p,
                               revocacao=r, pred=npred, ref=nref,
                               inst=int((inst[pid].escore >= lim).sum())))
            marca = "  <- pipeline default" if lim == 200 else ""
            print(f"{lim:>7} {int((inst[pid].escore>=lim).sum()):>6} {npred:>6} "
                  f"{p:>6.1%} {r:>6.1%} {f1:>6.1%}{marca}")
    df = pd.DataFrame(linhas)

    print("\n=== THE NUMBER THAT COUNTS: threshold selected on one plot, measured on the other ===")
    print(f"{'selected on':>13} {'measured on':>10} {'thresh':>7} | {'F1':>7} {'prec':>7} {'recall':>7} "
          f"| {'F1 with 200':>11} {'gain':>7}")
    resumo = []
    for esc, med in [(PLOTS[0], PLOTS[1]), (PLOTS[1], PLOTS[0])]:
        g = df[df.parcela == esc]
        lim = int(g.loc[g.F1.idxmax(), "limiar"])
        f1, p, r, _, _ = mede(inst[med], med, lim, REF, ctr)
        f200, _, _, _, _ = mede(inst[med], med, 200, REF, ctr)
        resumo.append(dict(escolhido_em=esc, medido_em=med, limiar=lim, F1=f1,
                           precisao=p, revocacao=r, F1_com_200=f200, ganho=f1 - f200))
        print(f"{esc:>13} {med:>10} {lim:>7} | {f1:>6.1%} {p:>6.1%} {r:>6.1%} "
              f"| {f200:>10.1%} {(f1-f200)*100:>+6.1f}")
    dr = pd.DataFrame(resumo)
    print(f"\nmean of both directions: F1 {dr.F1.mean():.1%} against {dr.F1_com_200.mean():.1%} "
          f"with threshold 200, i.e. {dr.ganho.mean()*100:+.1f} points")

    print("\nFor comparison: SegmentAnyTree with AdaBN, same plots, F1 92.1%")
    if dr.F1.mean() > 0.921:
        print("  FF3D OVERTAKES SegmentAnyTree. The conclusion of the work changes.")
    elif dr.F1.mean() > 0.90:
        print("  FF3D draws level with SegmentAnyTree. Its advantage was largely the filter.")
    else:
        print("  SegmentAnyTree stays ahead, but the gap narrows.")

    df.to_csv(config.OUT_DIR / "limiar_ruido_ff3d.csv", index=False)
    dr.to_csv(config.OUT_DIR / "limiar_ruido_ff3d_validado.csv", index=False)
    print(f"\nwrote to {config.OUT_DIR}")


if __name__ == "__main__":
    main()
