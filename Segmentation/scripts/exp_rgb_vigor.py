#!/usr/bin/env python3
"""Route 3. Does the RGB of the point cloud add anything beyond age?

Return and intensity speak of geometry and of laser reflectance; RGB speaks of colour, and canopy
colour is a proxy for vigour and nutrition, i.e. for site quality. Site is orthogonal to age: two
stands planted on the same day on different soils reach different volumes, so this is the route with
the best chance of explaining the 19% that age leaves behind.

Indices computed on the normalised colour of canopy points, all classics of RGB-camera remote
sensing: ExG (excess green), GLI, VARI, NGRDI, plus brightness and its dispersion.

Same confounder as route 2, and stronger here. The L2 colour comes from an integrated camera, so it
depends on time of flight, solar angle and shadow. If an index correlates with the point density of
the plot (which varies 2.7x with flight geometry), it is measuring acquisition and not forest. The
confounder test is part of the protocol.

Run: PYTHONPATH=. python scripts/exp_rgb_vigor.py
"""
import os
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from greenvista import config  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LAZ_DIR = Path(os.environ.get("GREENVISTA_LAZ_DIR", REPO / "work" / "lazall"))
SHP = config.DATA / "2-shapes" / "Parcelas" / "parcelas.shp"
FEAT = config.OUT_DIR / "ff3d_crown_features.csv"
OUT = config.OUT_DIR / "rgb_vigor.csv"
RNG = np.random.default_rng(20260730)
N_PERM = 2000

PLOTS = [(1, 1), (1, 2), (2, 6), (2, 7), (4, 4), (4, 5), (5, 8), (5, 9),
         (6, 10), (6, 11), (6, 12), (7, 13), (7, 14)]

g = gpd.read_file(SHP).to_crs(config.CRS)
cen = {(int(r.talhao), int(r.parcela)): (r.geometry.centroid.x, r.geometry.centroid.y)
       for r in g.itertuples()}
R_PLOT = config.PLOT_RADIUS_M

linhas = []
for talhao in sorted({t for t, _ in PLOTS}):
    laz = LAZ_DIR / f"SaoManuelTotal_{talhao:03d}.laz"
    if not laz.exists():
        continue
    las = laspy.read(str(laz))
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    rr = np.asarray(las.red).astype(np.float32)
    gg = np.asarray(las.green).astype(np.float32)
    bb = np.asarray(las.blue).astype(np.float32)
    print(f"stand {talhao}: {len(x):,} points", flush=True)
    for t, p in PLOTS:
        if t != talhao:
            continue
        cx, cy = cen[(t, p)]
        sel = ((x - cx) ** 2 + (y - cy) ** 2 <= R_PLOT ** 2) & \
              (z >= config.Z_MIN_M) & (z <= config.Z_MAX_M)
        r_, g_, b_ = rr[sel], gg[sel], bb[sel]
        val = (r_ + g_ + b_) > 0                      # drop the ~1% with no colour
        r_, g_, b_ = r_[val], g_[val], b_[val]
        s = r_ + g_ + b_
        rn, gn, bn = r_ / s, g_ / s, b_ / s           # normalised colour, removes brightness
        exg = 2 * gn - rn - bn
        gli = (2 * g_ - r_ - b_) / np.maximum(2 * g_ + r_ + b_, 1)
        vari = (g_ - r_) / np.maximum(g_ + r_ - b_, 1)
        ngrdi = (g_ - r_) / np.maximum(g_ + r_, 1)
        linhas.append(dict(
            chave=f"{t}_{p}", talhao=t, n_pts=int(sel.sum()),
            dens=float(sel.sum()) / (np.pi * R_PLOT ** 2),
            exg=float(exg.mean()), exg_dp=float(exg.std()),
            gli=float(gli.mean()), vari=float(np.clip(vari, -2, 2).mean()),
            ngrdi=float(ngrdi.mean()),
            verde_norm=float(gn.mean()), verm_norm=float(rn.mean()),
            brilho=float(s.mean() / 3), brilho_dp=float((s / 3).std())))
        print(f"  plot {p}: {int(val.sum()):,} canopy points with colour", flush=True)
    del las, x, y, z, rr, gg, bb

d = pd.DataFrame(linhas).sort_values("chave").reset_index(drop=True)
base = pd.read_csv(FEAT)
base["chave"] = base.talhao.astype(int).astype(str) + "_" + base.parcela.astype(int).astype(str)
d = d.merge(base[["chave", "Vol_ha", "idade_anos"]], on="chave")

COR = ["exg", "exg_dp", "gli", "vari", "ngrdi", "verde_norm", "verm_norm", "brilho", "brilho_dp"]
y = d.Vol_ha.to_numpy(float)
grp = d.talhao.to_numpy()
idade = d[["idade_anos"]].to_numpy(float)


def loto(X, t):
    p = np.empty(len(t))
    for gg_ in np.unique(grp):
        tr, te = grp != gg_, grp == gg_
        p[te] = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[tr], t[tr]).predict(X[te])
    return p


def r2(t, p):
    return 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()


print(f"\n{len(d)} plots, {d.talhao.nunique()} stands\n")
print(d[["chave", "idade_anos", "Vol_ha", "exg", "gli", "ngrdi", "brilho"]].to_string(
    index=False, float_format=lambda v: f"{v:.3f}"))

# ---------------------------------------------------- confounder, before interpreting anything
print("\n--- confounder: does the index measure forest or acquisition? ---")
limpo = []
for c in COR:
    cd = np.corrcoef(d[c], d.dens)[0, 1]
    ci = np.corrcoef(d[c], d.idade_anos)[0, 1]
    marca = "  <<< CONFOUNDED" if abs(cd) > 0.4 else ""
    if abs(cd) <= 0.4:
        limpo.append(c)
    print(f"  {c:<12} corr density {cd:+.3f}   corr age {ci:+.3f}{marca}")

# ---------------------------------------------------- R² under LOTO
print("\n--- accuracy on a stand the model never saw ---")
for nome, X in (("age only", idade),
                ("colour only (all)", d[COR].to_numpy(float)),
                ("colour only (unconfounded)", d[limpo].to_numpy(float) if limpo else None),
                ("age + clean colour",
                 np.column_stack([idade, d[limpo].to_numpy(float)]) if limpo else None)):
    if X is None:
        print(f"  {nome:<28} no clean index")
        continue
    p = loto(X, y)
    print(f"  {nome:<28} R² {r2(y, p):+.3f}   rRMSE "
          f"{100 * np.sqrt(((y - p) ** 2).mean()) / y.mean():5.1f}%")

# ---------------------------------------------------- orthogonalised residual
print("\n--- does colour predict the ERROR the age model makes? ---")
resid = y - loto(idade, y)
saida = []
for nome, cols in (("cor, todos os indices", COR),
                   ("cor, so os nao contaminados", limpo),
                   ("so a densidade de pontos", ["dens"])):
    if not cols:
        continue
    X = d[cols].to_numpy(float)
    pr = loto(X, resid)
    r_obs = np.corrcoef(resid, pr)[0, 1]
    nulo = np.array([np.corrcoef(resid, loto(X[RNG.permutation(len(y))], resid))[0, 1]
                     for _ in range(N_PERM)])
    p_val = float((np.abs(nulo) >= abs(r_obs)).mean())
    print(f"  {nome:<30} r {r_obs:+.3f}   R² {r2(resid, pr):+.3f}   "
          f"null sd {nulo.std():.3f}   p {p_val:.3f}")
    saida.append(dict(bloco=nome, r_obs=r_obs, R2_resid=r2(resid, pr),
                      nulo_dp=nulo.std(), p_perm=p_val))

pd.DataFrame(saida).to_csv(OUT, index=False)
d.to_csv(OUT.with_name("rgb_vigor_per_plot.csv"), index=False)
print(f"\nwrote {OUT}")
