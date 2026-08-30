#!/usr/bin/env python3
"""Count empty nodes of the planting lattice, instead of detecting trees.

Stem density ranges from 1025 to 2175 per hectare across the plots, a factor of two, the result of
mortality and planting failure. The planting date does not predict this, because two plots planted on
the same day may have lost very different numbers of trees; if the failure is measurable from the
flight, it is information orthogonal to age.

The task is potentially easier than segmentation: one is not looking for an unknown tree among
overlapping crowns, but for the occupancy of known positions, and the failure is exactly what shows
from above, because where there is no tree there is no crown to block the view.

Context of what has already been measured: `exp_grid_reconstruction.py` showed that the lattice
geometry is recoverable (azimuth ~4 degrees, spacing 2.68 against a nominal 2.87 m), while
`exp_grid_registration.py` and `exp_gps_ruler.py` showed that matching tree to tree fails because of a
limitation of the reference: the 25 GPS-surveyed trees fit the known lattice as badly as random points
(0.52 against 0.54 m). This experiment does not try to match tree to tree, it only counts how many
nodes are occupied, which is an aggregate question and does not require a correct position per
individual.

Estimate: nominal planting density (from the spacing in the inventory) x fraction of occupied nodes.

Run: PYTHONPATH=. python scripts/exp_lattice_gaps.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from greenvista import config  # noqa: E402
from greenvista.segmentation.ff3d import load_panoptic_ply  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PLY_DIR = REPO / "work" / "ff3d_degraded_out" / "full"
INV = config.DATA / "5-dados_campo" / "inv_euc.csv"
FEAT = config.OUT_DIR / "ff3d_crown_features.csv"
OUT = config.OUT_DIR / "lattice_gaps.csv"
RNG = np.random.default_rng(20260730)
N_PERM = 2000


def espac_para_m(s):
    """'2,87 x 1,88' or '1.52 x 3,33' -> (2.87, 1.88). The inventory mixes comma and dot."""
    if not isinstance(s, str):
        return None
    n = re.findall(r"\d+[.,]?\d*", s.replace(",", "."))
    return (float(n[0]), float(n[1])) if len(n) >= 2 else None


def centroides(ply):
    t = load_panoptic_ply(ply)
    # the loader returns {id: {"points": (N,3), "semantic": (N,)}}, not the array directly
    c = [v["points"][:, :2].mean(0) for v in t.values() if len(v["points"]) >= 30]
    return np.array(c) if c else np.zeros((0, 2))


def custo(par, pts, sx, sy):
    """Robust distance from each crown to the nearest lattice node, given (theta, x0, y0)."""
    th, x0, y0 = par
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    q = (pts - np.array([x0, y0])) @ R.T
    dx = q[:, 0] - np.round(q[:, 0] / sx) * sx
    dy = q[:, 1] - np.round(q[:, 1] / sy) * sy
    d = np.hypot(dx, dy)
    return np.mean(np.minimum(d, 0.6 * min(sx, sy)))          # simple Huber


def ajusta_grade(pts, sx, sy):
    melhor = None
    for th0 in np.linspace(0, np.pi / 2, 24, endpoint=False):
        for x0 in np.linspace(0, sx, 5, endpoint=False):
            for y0 in np.linspace(0, sy, 5, endpoint=False):
                c = custo((th0, x0, y0), pts, sx, sy)
                if melhor is None or c < melhor[0]:
                    melhor = (c, (th0, x0, y0))
    r = minimize(custo, melhor[1], args=(pts, sx, sy), method="Nelder-Mead",
                 options=dict(xatol=1e-3, fatol=1e-4, maxiter=2000))
    return r.x, r.fun


def ocupacao(pts, par, sx, sy):
    """Fraction of lattice nodes with a crown closer than half a spacing."""
    th, x0, y0 = par
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    q = (pts - np.array([x0, y0])) @ R.T
    # nodes that fall inside the envelope of the crowns
    i0, i1 = np.floor(q[:, 0].min() / sx), np.ceil(q[:, 0].max() / sx)
    j0, j1 = np.floor(q[:, 1].min() / sy), np.ceil(q[:, 1].max() / sy)
    nos = np.array([(i * sx, j * sy) for i in np.arange(i0, i1 + 1) for j in np.arange(j0, j1 + 1)])
    if not len(nos):
        return np.nan, 0, 0
    d = np.hypot(nos[:, None, 0] - q[None, :, 0], nos[:, None, 1] - q[None, :, 1]).min(1)
    ocup = d <= 0.5 * min(sx, sy)
    return float(ocup.mean()), int(ocup.sum()), len(nos)


# ------------------------------------------------------------------ field
inv = pd.read_csv(INV, sep=None, engine="python")
viv = inv[inv.D_cm.notna()]
campo = viv.groupby(["talhao", "parcela"]).agg(
    fustes=("arvore", "size"), area_m2=("area_parc", "first"),
    espac=("espac", lambda s: next((v for v in s if isinstance(v, str)), None))).reset_index()
campo["fustes_ha"] = campo.fustes / (campo.area_m2 / 1e4)
campo["chave"] = campo.talhao.astype(str) + "_" + campo.parcela.astype(str)

linhas = []
for _, r in campo.iterrows():
    ply = PLY_DIR / f"t{int(r.talhao):03d}_p{int(r.parcela):03d}_round2.ply"
    sp = espac_para_m(r.espac)
    if not ply.exists() or sp is None:
        print(f"  skipping {r.chave} (ply={ply.exists()}, spacing={r.espac})")
        continue
    sx, sy = sorted(sp, reverse=True)
    pts = centroides(ply)
    if len(pts) < 20:
        print(f"  skipping {r.chave}, only {len(pts)} crowns")
        continue
    par, resid = ajusta_grade(pts, sx, sy)
    # null for the lock-on: the same cost with the crowns shuffled inside the envelope
    lo, hi = pts.min(0), pts.max(0)
    nulo = [ajusta_grade(RNG.uniform(lo, hi, size=pts.shape), sx, sy)[1] for _ in range(12)]
    frac, nocup, ntot = ocupacao(pts, par, sx, sy)
    dens_nominal = 1e4 / (sx * sy)
    linhas.append(dict(chave=r.chave, talhao=int(r.talhao), copas=len(pts),
                       sx=sx, sy=sy, dens_nominal=dens_nominal,
                       resid_ajuste=resid, nulo_ajuste=float(np.mean(nulo)),
                       lock_on=float(np.mean(nulo)) / resid,
                       ocupacao=frac, nos_ocupados=nocup, nos_total=ntot,
                       dens_estimada=dens_nominal * frac, dens_campo=r.fustes_ha))
    print(f"  {r.chave}: {len(pts)} crowns, lock-on {resid:.3f} m against null "
          f"{np.mean(nulo):.3f} ({np.mean(nulo)/resid:.2f}x), occupancy {frac:.2f}", flush=True)

d = pd.DataFrame(linhas)
print("\n" + d[["chave", "copas", "lock_on", "ocupacao", "dens_nominal",
                "dens_estimada", "dens_campo"]].to_string(index=False,
                                                          float_format=lambda v: f"{v:.2f}"))

if len(d) >= 6:
    r_dens = np.corrcoef(d.dens_estimada, d.dens_campo)[0, 1]
    vies = 100 * (d.dens_estimada.mean() / d.dens_campo.mean() - 1)
    print(f"\nestimated density against field: r = {r_dens:+.3f}, bias {vies:+.1f}%")
    print(f"mean lattice lock-on: {d.lock_on.mean():.2f}x better than random crowns")

    # does the estimated density predict the error the age model makes?
    base = pd.read_csv(FEAT)
    base["chave"] = base.talhao.astype(int).astype(str) + "_" + base.parcela.astype(int).astype(str)
    m = d.merge(base[["chave", "Vol_ha", "idade_anos"]], on="chave")
    y, g = m.Vol_ha.to_numpy(float), m.talhao.to_numpy()

    def loto(X, t):
        p = np.empty(len(t))
        for gg in np.unique(g):
            tr, te = g != gg, g == gg
            p[te] = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[tr], t[tr]).predict(X[te])
        return p

    def r2(t, p):
        return 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()

    resid = y - loto(m[["idade_anos"]].to_numpy(float), y)
    for nome, cols in (("estimated density", ["dens_estimada"]),
                       ("lattice occupancy", ["ocupacao"]),
                       ("FIELD density (ceiling)", ["dens_campo"])):
        X = m[cols].to_numpy(float)
        pr = loto(X, resid)
        r_obs = np.corrcoef(resid, pr)[0, 1]
        nulo = np.array([np.corrcoef(resid, loto(X[RNG.permutation(len(y))], resid))[0, 1]
                         for _ in range(N_PERM)])
        print(f"  residual ~ {nome:<26} r {r_obs:+.3f}  R² {r2(resid, pr):+.3f}  "
              f"p {(np.abs(nulo) >= abs(r_obs)).mean():.3f}")

d.to_csv(OUT, index=False)
print(f"\nwrote {OUT}")
