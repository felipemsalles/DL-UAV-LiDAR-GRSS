#!/usr/bin/env python3
"""How much of the matching is real skill and how much is the chance the threshold allows.

Matched recall at 2 m is reported without a comparison floor: what is missing is the
recall of a detector that detects nothing. In a plantation with 1.88 m spacing, and
with more detections than stems, a 2 m threshold is generous, because almost every
stem has something within 2 m of it. Without the chance floor, the reported number
does not state the size of the skill.

The null is 1076 points thrown uniformly over the same extent, i.e. the same number
of detections the method produces, with no information from the point cloud at all.
Whatever it reaches is what the threshold gives away for free.

The result changes how the number reads: at 2 m chance alone already reaches 82%, so
the 96.9% sit 15 points above it; at 1 m chance reaches 40% and the method reaches
78%, a margin of 37 points. The evidence of real detection lies at the tight
threshold, not the loose one, even though the absolute number there is lower.

Run: PYTHONPATH=. python scripts/exp_nulo_pareamento.py
Output: manual_match/nulo_pareamento_talhao001.csv
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenvista import config  # noqa: E402

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DETEC = config.REPO / "data/detections/sat_w2w_arvores.csv"
SAIDA = config.OUT_DIR / "nulo_pareamento_talhao001.csv"
LIMIARES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
SORTEIOS = 15
SEMENTE = 11        # fixed, otherwise the null changes every run and the table does not reconcile


def revocacao(ref, pred, lim):
    """Recall, precision and F1 of the one-to-one matching, plus the pair distances.

    Precision and F1 are not independent evidence here, because of how the null is
    designed: it throws exactly as many points as the method detects, so the
    precision denominator is the same on both sides and P = R x len(ref)/len(pred)
    for both. The F1 margin is a monotone transform of the recall margin. All three
    are reported anyway, so that they are recorded in the CSV.
    """
    D = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    li, ci = linear_sum_assignment(np.where(D <= lim, D, 1e6))
    ok = D[li, ci] <= lim
    tp = int(ok.sum())
    r = 100 * tp / len(ref)
    pr = 100 * tp / len(pred)
    f1 = 2 * pr * r / (pr + r) if pr + r else 0.0
    return r, pr, f1, D[li, ci][ok]


def main():
    rng = np.random.default_rng(SEMENTE)
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    d = pd.read_csv(DETEC)
    pred = np.column_stack([d[d.talhao == 1].base_x.values, d[d.talhao == 1].base_y.values])
    lo, hi = ref.min(0), ref.max(0)

    print(f"{len(ref)} stems, {len(pred)} detections, {SORTEIOS} draws per threshold\n")
    cab = (f"{'thresh':>7} | {'R meth':>6} {'R chn':>6} {'marg':>6} | "
           f"{'P meth':>6} {'P chn':>6} {'marg':>6} | {'F1 meth':>7} {'F1 chn':>7} {'marg':>6}")
    print(cab); print("-" * len(cab))
    linhas = []
    for lim in LIMIARES:
        nosso, p_nosso, f_nosso, dist = revocacao(ref, pred, lim)
        sorteios = [revocacao(ref, np.column_stack([
            rng.uniform(lo[0], hi[0], len(pred)),
            rng.uniform(lo[1], hi[1], len(pred))]), lim)[:3] for _ in range(SORTEIOS)]
        m, s = float(np.mean([x[0] for x in sorteios])), float(np.std([x[0] for x in sorteios]))
        pm = float(np.mean([x[1] for x in sorteios]))
        fm = float(np.mean([x[2] for x in sorteios]))
        linhas.append({"limiar_m": lim, "revocacao_metodo_pct": round(nosso, 1),
                       "revocacao_acaso_pct": round(m, 1), "acaso_dp": round(s, 1),
                       "margem_pp": round(nosso - m, 1),
                       "precisao_metodo_pct": round(p_nosso, 1),
                       "precisao_acaso_pct": round(pm, 1),
                       "margem_precisao_pp": round(p_nosso - pm, 1),
                       "f1_metodo_pct": round(f_nosso, 1),
                       "f1_acaso_pct": round(fm, 1),
                       "margem_f1_pp": round(f_nosso - fm, 1),
                       "dist_mediana_m": round(float(np.median(dist)), 2) if len(dist) else None})
        print(f"{lim:>6.2f}m | {nosso:>5.1f}% {m:>5.1f}% {nosso-m:>5.1f} | "
              f"{p_nosso:>5.1f}% {pm:>5.1f}% {p_nosso-pm:>5.1f} | "
              f"{f_nosso:>6.1f}% {fm:>6.1f}% {f_nosso-fm:>5.1f}")

    *_, dist = revocacao(ref, pred, 2.0)
    print(f"\npair distance at the 2 m threshold: median {np.median(dist):.2f} m, "
          f"{100 * (dist < 1).mean():.0f}% below 1 m")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(SAIDA, index=False)
    print(SAIDA)


if __name__ == "__main__":
    main()
