#!/usr/bin/env python3
"""Can the oblique return at breast height be used to confirm a detection?

`exp_dap_obliquo_agregado.py` tried to measure DBH on the oblique ring and the control
rejected it: at an empty position, more than 80 cm from any stem, the estimator still returns
a number, and it returns 26 cm. As a diameter measurement, it is useless.

The same control showed that the estimator resolves 55 % of the true stems and only 3.7 % of
the empty positions. What it carries is stem presence, not diameter, and stem presence is
exactly what is missing to separate a good detection from a false positive.

`exp_por_arvore_com_809_rotulos` showed that within a stand volume tracks stem count
(r = +0.864), not stem size, so improving the count improves the estimate that matters.

The test is against the TLS map: the `casou` column of `deteccoes_julgadas_talhao001.csv`
carries the matching validated with a 2 m tolerance, the same as in the paper. There is no new
judgement here, only a new feature tested against a label that already existed.

Control: the same feature computed at a randomly drawn position. If it separates good from bad
detections much less than it separates stem from empty ground, then what it measures is
"there is something here" and not "there is a stem here".

Run: PYTHONPATH=. python scripts/exp_obliquo_filtra_deteccao.py
Output: manual_match/obliquo_filtra_deteccao.csv
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402

_s = importlib.util.spec_from_file_location("obl", R / "scripts/exp_dap_do_obliquo.py")
_o = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_o)
warnings.filterwarnings("ignore")

DET = config.OUT_DIR / "deteccoes_julgadas_talhao001.csv"
DRONE = config.REPO / "work/lazall/SaoManuelTotal_001.laz"
SAIDA = config.OUT_DIR / "obliquo_filtra_deteccao.csv"
TRECHO = (1.0, 6.0)
FAIXAS = [(0, 15), (20, 36), (0, 36)]
RAIOS = [0.30, 0.45, 0.60]
SEED = 20260828


def evidencia(X, Y, pos, raio):
    """Stem evidence under each position: count, concentration and whether the ring resolves."""
    T = cKDTree(np.column_stack([X, Y]))
    n = np.zeros(len(pos))
    conc = np.full(len(pos), np.nan)
    resolve = np.zeros(len(pos), bool)
    for i, (cx, cy) in enumerate(pos):
        idx = np.asarray(T.query_ball_point([cx, cy], raio), dtype=int)
        n[i] = len(idx)
        if len(idx) >= 5:
            d = np.hypot(X[idx] - cx, Y[idx] - cy)
            # concentration against the uniform null: fraction within 20 cm over (0.20/radius)^2
            conc[i] = np.mean(d < 0.20) / ((0.20 / raio) ** 2)
        if len(idx) >= _o.PTS_MIN:
            r, _, _ = _o.raio_por_modo(X[idx] - cx, Y[idx] - cy)
            resolve[i] = np.isfinite(r)
    return n, conc, resolve


def main():
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(DET)
    d = d[d.talhao == 1].copy()
    pos = d[["base_x", "base_y"]].to_numpy(float)
    alvo = d.casou.astype(bool).to_numpy()
    print(f"{len(d)} detections in stand 001, {alvo.sum()} matched to the TLS map "
          f"({100 * alvo.mean():.1f} %)\n")

    las = laspy.read(str(DRONE))
    cls = np.asarray(las.classification)
    veg = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    x, y, z = np.asarray(las.x)[veg], np.asarray(las.y)[veg], np.asarray(las.z)[veg]
    ang = np.abs(np.asarray(las.scan_angle_rank).astype(float))[veg]

    # control: positions drawn at random over the same area
    falsas = np.column_stack([rng.uniform(pos[:, 0].min(), pos[:, 0].max(), len(pos)),
                              rng.uniform(pos[:, 1].min(), pos[:, 1].max(), len(pos))])

    linhas = []
    print(f"{'band':>10} {'radius':>6} {'AUC n':>8} {'AUC conc':>9} {'AUC resolve':>12} "
          f"{'resolve matched':>15} {'resolve not':>12} {'resolve random':>17}")
    for a0, a1 in FAIXAS:
        m = (z >= TRECHO[0]) & (z <= TRECHO[1]) & (ang >= a0) & (ang < a1)
        if m.sum() < 5000:
            continue
        for raio in RAIOS:
            n, conc, res = evidencia(x[m], y[m], pos, raio)
            _, _, res_f = evidencia(x[m], y[m], falsas, raio)
            ok = np.isfinite(conc)
            auc_n = roc_auc_score(alvo, n)
            auc_c = roc_auc_score(alvo[ok], conc[ok]) if ok.sum() > 50 else np.nan
            auc_r = roc_auc_score(alvo, res.astype(float))
            print(f"{a0:4d}-{a1:3d}  {raio:6.2f} {auc_n:8.3f} {auc_c:9.3f} {auc_r:12.3f} "
                  f"{100*res[alvo].mean():14.1f}% {100*res[~alvo].mean():11.1f}% "
                  f"{100*res_f.mean():16.1f}%", flush=True)
            linhas.append(dict(a0=a0, a1=a1, raio=raio, auc_n=auc_n, auc_conc=auc_c,
                               auc_resolve=auc_r,
                               resolve_casada=float(res[alvo].mean()),
                               resolve_nao_casada=float(res[~alvo].mean()),
                               resolve_sorteada=float(res_f.mean())))

    d2 = pd.DataFrame(linhas)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    d2.to_csv(SAIDA, index=False)
    if len(d2):
        b = d2.loc[d2.auc_conc.idxmax()]
        print(f"\nBEST separator: band {b.a0:.0f}-{b.a1:.0f}, radius {b.raio:.2f} m, "
              f"AUC {b.auc_conc:.3f}")
        print("   An AUC of 0.5 is chance. Above 0.65 it is already useful as a filter; below "
              "0.6 it supports nothing.")
        print(f"   and the control: at a random position the ring resolves in "
              f"{100*b.resolve_sorteada:.1f} % of cases against {100*b.resolve_casada:.1f} % at matched ones.")
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()
