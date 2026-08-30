#!/usr/bin/env python3
"""Route 2. Do return and intensity metrics add anything beyond age?

All ~40 metrics in `greenvista/area_based/data.py` are height metrics (zmax, zmean, zq*, zpcum*,
zsd, zskew, zkurt, zentropy, pzabovezmean). The data is point format 3 with up to 5 returns per
pulse, `intensity` and RGB, and none of that has entered a model so far. Return structure speaks
of foliage density and of the crown interior, which is different information from where the top
is: two plots can share the same height profile and have very different crowns inside.

Return-number statistics come first because they are robust. `intensity` is usually poorly
calibrated across flight lines, so it enters separately and its result carries a caveat.

The main test is the orthogonalised-residual one, as in `lidar_beyond_age.csv`: fit with age
alone, take what is left, and ask whether the new metrics predict that residual. Comparing the R2
of models with different numbers of variables on 13 samples rewards whichever has more variables,
not whichever has more information. The permutation null is necessary: with 6 stands the null
distribution of r is wide (standard deviation ~0.5 in the previous experiment), so a correlation
of 0.3 means nothing.

Run: PYTHONPATH=. python scripts/exp_return_intensity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from greenvista import config  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "work" / "plot_points_cache.npz"
FEAT = config.OUT_DIR / "ff3d_crown_features.csv"
OUT = config.OUT_DIR / "return_intensity.csv"
RNG = np.random.default_rng(20260730)
N_PERM = 2000


def metricas(xyz, inten, rnum, nret, cls):
    """Return and intensity metrics over the same height window as the height metrics."""
    z = xyz[:, 2]
    dossel = (z >= config.Z_MIN_M) & (z <= config.Z_MAX_M)
    d = {}
    if dossel.sum() < 100:
        return None
    rn, nr = rnum[dossel].astype(float), nret[dossel].astype(float)
    d["prop_pulso_unico"] = float((nr == 1).mean())        # pulse that did not penetrate
    d["prop_primeiro"] = float((rn == 1).mean())
    d["prop_ultimo"] = float((rn == nr).mean())
    d["prop_intermediario"] = float(((rn > 1) & (rn < nr)).mean())
    d["nret_medio"] = float(nr.mean())
    d["nret_dp"] = float(nr.std())
    # real penetration measured by the return, not by height
    d["prop_retorno_solo"] = float((cls == 2).mean())
    d["razao_ultimo_primeiro"] = d["prop_ultimo"] / max(d["prop_primeiro"], 1e-6)

    it = inten[dossel].astype(float)
    d["int_media"] = float(it.mean())
    d["int_dp"] = float(it.std())
    d["int_q90"] = float(np.percentile(it, 90))
    solo = cls == 2
    d["int_razao_dossel_solo"] = (float(it.mean() / max(inten[solo].astype(float).mean(), 1e-6))
                                  if solo.sum() > 50 else np.nan)
    return d


def loto_pred(X, y, grupos, alpha=1.0):
    """Leave-one-STAND-out prediction. 6 folds, because there are 6 stands."""
    pred = np.empty_like(y, dtype=float)
    for g in np.unique(grupos):
        tr, te = grupos != g, grupos == g
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# ------------------------------------------------------------------ data
z = np.load(CACHE)
base = pd.read_csv(FEAT)
base["chave"] = base.talhao.astype(int).astype(str) + "_" + base.parcela.astype(int).astype(str)

linhas = []
for _, r in base.iterrows():
    k = r["chave"]
    if k + "|xyz" not in z:
        continue
    m = metricas(z[k + "|xyz"], z[k + "|inten"], z[k + "|rnum"], z[k + "|nret"], z[k + "|cls"])
    if m is None:
        continue
    m.update(chave=k, talhao=int(r.talhao), Vol_ha=float(r.Vol_ha), idade=float(r.idade_anos))
    linhas.append(m)

d = pd.DataFrame(linhas).sort_values("chave").reset_index(drop=True)
RET = ["prop_pulso_unico", "prop_primeiro", "prop_ultimo", "prop_intermediario",
       "nret_medio", "nret_dp", "prop_retorno_solo", "razao_ultimo_primeiro"]
INT = ["int_media", "int_dp", "int_q90", "int_razao_dossel_solo"]
d[INT] = d[INT].fillna(d[INT].mean())

y = d.Vol_ha.to_numpy(float)
g = d.talhao.to_numpy()
idade = d[["idade"]].to_numpy(float)

print(f"{len(d)} plots, {d.talhao.nunique()} stands\n")
print(d[["chave", "idade", "Vol_ha", "prop_pulso_unico", "nret_medio", "prop_retorno_solo",
         "int_media"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ------------------------------------------------------------------ 1. R2 under LOTO
print("\n--- accuracy on a stand the model never saw (leave-one-stand-out R²) ---")
combos = [("so idade", idade),
          ("so retorno", d[RET].to_numpy(float)),
          ("so intensidade", d[INT].to_numpy(float)),
          ("idade + retorno", np.column_stack([idade, d[RET].to_numpy(float)])),
          ("idade + retorno + intensidade",
           np.column_stack([idade, d[RET].to_numpy(float), d[INT].to_numpy(float)]))]
res = {}
for nome, X in combos:
    p = loto_pred(X, y, g)
    res[nome] = (r2(y, p), 100 * np.sqrt(((y - p) ** 2).mean()) / y.mean())
    print(f"  {nome:<30} R² {res[nome][0]:+.3f}   rRMSE {res[nome][1]:5.1f}%")

# ------------------------------------------------------------------ 2. orthogonalised residual
print("\n--- does the return signal predict the ERROR the age model makes? ---")
resid = y - loto_pred(idade, y, g)
saida = []
for nome, cols in (("retorno", RET), ("intensidade", INT), ("retorno + intensidade", RET + INT)):
    X = d[cols].to_numpy(float)
    pr = loto_pred(X, resid, g)
    r_obs = np.corrcoef(resid, pr)[0, 1]
    nulo = []
    for _ in range(N_PERM):
        perm = RNG.permutation(len(y))
        nulo.append(np.corrcoef(resid, loto_pred(X[perm], resid, g))[0, 1])
    nulo = np.array(nulo)
    p_val = float((np.abs(nulo) >= abs(r_obs)).mean())
    print(f"  {nome:<24} r {r_obs:+.3f}   R² {r2(resid, pr):+.3f}   "
          f"null {nulo.mean():+.3f} sd {nulo.std():.3f}   p {p_val:.3f}")
    saida.append(dict(bloco=nome, r_obs=r_obs, R2_resid=r2(resid, pr),
                      nulo_media=nulo.mean(), nulo_dp=nulo.std(), p_perm=p_val))

pd.DataFrame(saida).to_csv(OUT, index=False)
d.to_csv(OUT.with_name("return_intensity_per_plot.csv"), index=False)
print(f"\nwrote {OUT}")
