"""Fusion radius selected outside the set on which it is measured.

The first radius sweep (2026-08-15) picked the optimum by looking at stand 001
itself, which is where the stem map lies, and that inflates both models. Here the
radius is selected on one plot and measured on the other, in both directions.

All previous comparisons used 1.5 m for both models, and 1.5 m is the optimum for
SegmentAnyTree but not for FF3D, which prefers 1.0 to 1.2 m; FF3D was therefore
being compared at a disadvantage of about 3.4 F1 points.

With only two plots, cross-validation reduces to selecting on one and measuring on
the other, and the estimate is noisy. What remains solid is the direction: the two
models have different optima, and using a single radius penalised one of them.

Run: PYTHONPATH=. python scripts/exp_raio_fusao_validado.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_tta_comparison import MAPA, PLOTS, centros_e_censo, no_quadrado, parear  # noqa: E402
from ff3d_detection_overlap import nms_merge  # noqa: E402

from greenvista import config  # noqa: E402

RAIOS = [0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.1, 2.5]
LIMIAR = 2.0
FONTE = config.REPO / "data/detections/tta_t001_instancias.csv"


def marca(sub, pid, raio, ref, ctr):
    """(TP, FP, FN) for one plot, at a given fusion radius."""
    s = sub[sub.plot_id == pid]
    cx, cy, _ = ctr[pid]
    pred = no_quadrado(nms_merge(s[["cx", "cy"]].to_numpy(float),
                                 s.n_pts.to_numpy(float), raio), cx, cy)
    r = no_quadrado(ref, cx, cy)
    tp, _ = parear(r, pred, LIMIAR)
    return tp, len(pred) - tp, len(r) - tp


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0, p, r


def main():
    d = pd.read_csv(FONTE)
    ref_all = gpd.read_file(MAPA)
    REF = np.column_stack([ref_all.geometry.x, ref_all.geometry.y])
    ctr = centros_e_censo()

    linhas = []
    for (modelo, cond, pos), sub in d.groupby(["modelo", "condicao", "posicao"]):
        for escolha, medida in [(PLOTS[0], PLOTS[1]), (PLOTS[1], PLOTS[0])]:
            # pick the radius looking only at the selection plot
            melhor_r, melhor_f = None, -1.0
            for raio in RAIOS:
                v = f1(*marca(sub, escolha, raio, REF, ctr))[0]
                if v > melhor_f:
                    melhor_f, melhor_r = v, raio
            # measure on the other plot, with the radius already fixed
            fv, p, r = f1(*marca(sub, medida, melhor_r, REF, ctr))
            # and what a fixed 1.5 m would give on that same measurement plot
            f15, _, _ = f1(*marca(sub, medida, 1.5, REF, ctr))
            linhas.append(dict(modelo=modelo, condicao=cond, posicao=pos,
                               escolhido_em=escolha, medido_em=medida,
                               raio=melhor_r, F1=fv, precisao=p, revocacao=r,
                               F1_com_1_5=f15, ganho=fv - f15))
    df = pd.DataFrame(linhas)

    print(f"radius selected on one plot, F1 measured on the OTHER, threshold {LIMIAR} m\n")
    print(f"{'model':>15} {'cond':>17} {'pos':>10} {'sel':>5}->{'meas':<5} "
          f"{'radius':>5} {'F1':>7} {'F1 at 1.5':>9} {'gain':>7}")
    for _, r in df.sort_values(["modelo", "condicao", "posicao"]).iterrows():
        print(f"{r.modelo:>15} {r.condicao:>17} {r.posicao:>10} {r.escolhido_em:>5}->{r.medido_em:<5} "
              f"{r.raio:>5.1f} {r.F1:>6.1%} {r.F1_com_1_5:>8.1%} {r.ganho*100:>+6.1f}")

    print("\n=== mean of both directions, i.e. the out-of-fold estimate ===")
    m = df.groupby(["modelo", "condicao", "posicao"]).agg(
        raio=("raio", "mean"), F1=("F1", "mean"),
        F1_com_1_5=("F1_com_1_5", "mean"), ganho=("ganho", "mean")).reset_index()
    for _, r in m.iterrows():
        print(f"{r.modelo:>15} {r.condicao:>17} {r.posicao:>10} | mean radius {r.raio:>4.2f} | "
              f"F1 {r.F1:>6.1%} against {r.F1_com_1_5:>6.1%} with a fixed 1.5 m | {r.ganho*100:>+5.1f}")

    out = config.OUT_DIR / "raio_fusao_validado.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
