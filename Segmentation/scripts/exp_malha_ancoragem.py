"""Feasibility of recovering the planting lattice from the detections.

Eucalyptus is planted on a regular lattice, 1.88 m within the row and 2.87 m between
rows. Neither segmenter uses that information. If the lattice can be recovered, it
covers both sides of the error: it fills the node where a tree should be and the
model found none, and it discards the detection that fell between nodes. This script
is the feasibility test, not the post-processing.

Difference with respect to `exp_grid_registration.py`: there the lattice comes from
the census, through the row/tree numbering, and so it only runs where a per-tree
census exists. Here azimuth, origin and phase come from the detections themselves,
which allows it to run over all 13 plots.

Control: a lattice with three free parameters fits any point cloud reasonably well,
so the real fit is compared with the fit of the same lattice to shuffled positions
inside the same area.

Run: PYTHONPATH=. python scripts/exp_malha_ancoragem.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_tta_comparison import RAIO_400, centros_e_censo  # noqa: E402
from ff3d_detection_overlap import nms_merge  # noqa: E402

from greenvista import config  # noqa: E402

S_LINHA, S_ENTRE = 1.88, 2.87     # nominal planting spacing, in metres
RAIO_FUSAO = 1.5
HUBER = 1.0                       # robust ceiling of the node -> detection cost
RAIO_AVAL = RAIO_400 + 4.0        # evaluate on a disc slightly larger than the plot
RNG = np.random.default_rng(0)
FONTE_SAT = config.REPO / "project/models/SegmentAnyTree_blackwell/bucket_out_adabn13"


def deteccoes_por_parcela(pasta):
    """Fused detections per plot, in UTM."""
    import laspy
    por = {}
    for f in sorted(Path(pasta).glob("*_out.laz")):
        t, p = int(f.name[1:4]), int(f.name.split("_p")[1][:3])
        las = laspy.read(str(f))
        inst = np.asarray(las.PredInstance)
        xy = np.column_stack([np.asarray(las.x), np.asarray(las.y)])
        cs, ss = [], []
        for i in np.unique(inst[inst > 0]):
            m = inst == i
            cs.append(xy[m].mean(0)); ss.append(int(m.sum()))
        d = por.setdefault(f"{t}_{p}", [[], []])
        d[0].extend(cs); d[1].extend(ss)
    return {k: nms_merge(np.asarray(v[0]).reshape(-1, 2), np.asarray(v[1]), RAIO_FUSAO)
            for k, v in por.items()}


def malha(cx, cy, raio, theta, dx, dy):
    """Nodes of a regular lattice covering the disc, at a given azimuth and phase."""
    n_l = int(np.ceil(2 * raio / S_LINHA)) + 2
    n_e = int(np.ceil(2 * raio / S_ENTRE)) + 2
    i, j = np.meshgrid(np.arange(-n_l, n_l + 1), np.arange(-n_e, n_e + 1))
    u, v = i.ravel() * S_LINHA + dx, j.ravel() * S_ENTRE + dy
    c, s = np.cos(theta), np.sin(theta)
    x, y = cx + u * c - v * s, cy + u * s + v * c
    d = np.hypot(x - cx, y - cy)
    m = d <= raio
    return np.column_stack([x[m], y[m]])


def custo(p, det, cx, cy, raio):
    """Robust cost: distance from each detection to the nearest node.

    The direction is detection -> node: an empty node is expected, because mortality
    and planting failure reach 25% in these plots. It is a detection far from any
    node that indicates an error.
    """
    theta, dx, dy = p
    nos = malha(cx, cy, raio + 3.0, theta, dx, dy)
    if len(nos) == 0:
        return 1e9
    d, _ = cKDTree(nos).query(det)
    return float(np.minimum(d, HUBER).mean())


def ajusta(det, cx, cy, raio):
    """Coarse search over azimuth and phase, then refinement."""
    melhor, arg = 1e9, None
    for theta in np.arange(0, np.pi, np.deg2rad(3)):
        for dx in np.arange(0, S_LINHA, S_LINHA / 4):
            for dy in np.arange(0, S_ENTRE, S_ENTRE / 4):
                c = custo((theta, dx, dy), det, cx, cy, raio)
                if c < melhor:
                    melhor, arg = c, (theta, dx, dy)
    r = minimize(custo, arg, args=(det, cx, cy, raio), method="Nelder-Mead",
                 options=dict(xatol=1e-3, fatol=1e-5, maxiter=800))
    return (r.x if r.fun < melhor else np.array(arg)), min(r.fun, melhor)


def main():
    ctr = centros_e_censo()
    print(f"nominal lattice {S_LINHA} x {S_ENTRE} m, i.e. "
          f"{1e4/(S_LINHA*S_ENTRE):.0f} trees/ha if none died\n")
    print("reading SegmentAnyTree detections with AdaBN, 13 plots")
    det_todas = deteccoes_por_parcela(FONTE_SAT)
    print(f"  {len(det_todas)} plots\n")

    linhas = []
    print(f"{'plot':>5} {'det':>4} {'census':>6} | {'azimuth':>8} {'cost':>7} "
          f"{'shuffled':>12} {'ratio':>6} | {'<0.5m':>7} {'<1.0m':>7}")
    for pid, det in sorted(det_todas.items()):
        cx, cy, campo = ctr[pid]
        d = det[np.hypot(det[:, 0] - cx, det[:, 1] - cy) <= RAIO_AVAL]
        if len(d) < 20:
            continue
        p, c = ajusta(d, cx, cy, RAIO_AVAL)

        # Control: the same number of points at random positions inside the same
        # disc, with the lattice refitted to them.
        cs = []
        for _ in range(3):
            ang = RNG.uniform(0, 2*np.pi, len(d))
            rad = RAIO_AVAL * np.sqrt(RNG.uniform(0, 1, len(d)))
            falso = np.column_stack([cx + rad*np.cos(ang), cy + rad*np.sin(ang)])
            cs.append(ajusta(falso, cx, cy, RAIO_AVAL)[1])
        c_falso = float(np.mean(cs))

        nos = malha(cx, cy, RAIO_AVAL + 3.0, *p)
        dist, _ = cKDTree(nos).query(d)
        linhas.append(dict(parcela=pid, n_det=len(d), campo=int(campo),
                           azimute_graus=float(np.rad2deg(p[0]) % 180), custo=c,
                           custo_embaralhado=c_falso, razao=c / c_falso,
                           frac_0_5=float((dist <= 0.5).mean()),
                           frac_1_0=float((dist <= 1.0).mean())))
        r = linhas[-1]
        print(f"{pid:>5} {len(d):>4} {int(campo):>6} | {r['azimute_graus']:>7.1f}° "
              f"{c:>7.3f} {c_falso:>12.3f} {r['razao']:>6.2f} | "
              f"{r['frac_0_5']:>6.1%} {r['frac_1_0']:>6.1%}")

    df = pd.DataFrame(linhas)
    print(f"\nmedian of the real cost / shuffled cost ratio: {df.razao.median():.2f}")
    print(f"median share of detections within 0.5 m of a node: {df.frac_0_5.median():.1%}")
    print(f"azimuth spread across plots of the same stand (expected: small):")
    df["talhao"] = df.parcela.str.split("_").str[0]
    for t, g in df.groupby("talhao"):
        if len(g) > 1:
            print(f"  stand {t}: {', '.join(f'{a:.1f}°' for a in g.azimute_graus)}")

    print("\nVERDICT")
    if df.razao.median() < 0.75:
        print("  The lattice is recoverable from the detections. The post-processing is worth writing.")
    elif df.razao.median() < 0.9:
        print("  Weak signal. The lattice shows up, but not enough to anchor on safely.")
    else:
        print("  NEGATIVE: the real fit does not beat the shuffled one. The idea dies here.")

    out = config.OUT_DIR / "malha_ancoragem.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
