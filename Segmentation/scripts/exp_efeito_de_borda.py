#!/usr/bin/env python3
"""How much of the per-plot error is edge effect, that is, comes from the reference itself.

Griese, Kleinn and Nolke (arXiv 2607.05260, 2026) argue that the per-plot field reference is
discrete, a tree counts in full if its stem falls inside and counts for nothing if it falls
outside, whereas the LiDAR sees what is above the plot, which is crown. Crown from a tree inside
goes out, crown from a tree outside comes in, and the difference is an error no model removes.
They measure at 100, 400, 900, 1600 and 2500 m2 and find a larger effect on the small plots: at
100 m2 a continuous reference cut 16.84 % of the RRMSE and added 0.22 of R2. The plots of this
project are 400 m2, inside the range where the effect still bites. The authors themselves note
that in a homogeneous plantation the effect should be smaller than in natural forest, and that is
why it is measured here instead of assumed.

This is not a new model, it is a floor: the question is not whether the model improves, but how
much of the reported error comes from the reference and not from the method. A reference floor
limits any model.

With uniform crowns the continuous reference is an exact smoothing of the discrete one: if every
crown were a disc of fixed radius Rc, then N_continuous(c) = mean of N_discrete over displacements
inside the disc, an identity and not an approximation, and the gain would be pure smoothing. That
is why the crown radius here scales with DBH, and fixed-kernel smoothing enters as a control: if
the control explains the gain, the gain was smoothing.

The crown radius is calibrated on the data itself and swept. The mean segmented crown area on the
two plots of stand 001 is 10.85 and 8.75 m2, that is, radii of 1.86 and 1.67 m; the coefficient
follows from that and is swept by plus or minus 25 %, so the conclusion does not depend on a
hand-picked constant.

An overlapping virtual plot is a pseudo-replicate. In the geometric part that does not matter,
because there is no model to leak. In the model part the plots are spaced by one diameter and the
validation is by spatial block with a guard band, otherwise a neighbouring plot trains its
neighbour and the RRMSE falls artificially.

Usage: PYTHONPATH=. python scripts/exp_efeito_de_borda.py
Output: manual_match/efeito_de_borda.csv and manual_match/efeito_de_borda_modelo.csv
"""
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenvista import config  # noqa: E402
from greenvista.area_based.mapping import stdmetrics_subset  # noqa: E402

warnings.filterwarnings("ignore")

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
REF_DAP = config.OUT_DIR / "dap_tls_ransac_talhao001.csv"
DRONE_T001 = config.REPO / "work/lazall/SaoManuelTotal_001.laz"
SAIDA = config.OUT_DIR / "efeito_de_borda.csv"
SAIDA_MOD = config.OUT_DIR / "efeito_de_borda_modelo.csv"

AREAS = (25.0, 50.0, 100.0, 200.0, 400.0, 900.0)
PASSO_GRADE = 1.0          # spacing of the centres in the geometric part, in metres
A_COPA = 0.0927            # crown radius per cm of DBH, calibrated on the segmented crowns
VARRE_COPA = (0.75, 1.0, 1.25)
LADO_QUADRADO = 26.0       # the evaluation square of the paper
RAIO_PARCELA = 11.28       # 400 m2 circle, which is the field plot
SEED = 20260828


def intersecao(d, r, R):
    """Intersection area of two circles, vectorised over d and r."""
    d = np.asarray(d, float)
    r = np.asarray(r, float)
    out = np.zeros(np.broadcast(d, r).shape)
    dentro = d <= np.abs(R - r)
    fora = d >= (r + R)
    meio = ~(dentro | fora)
    rr = np.broadcast_to(r, out.shape)
    out[dentro] = np.pi * np.minimum(rr[dentro], R) ** 2
    if meio.any():
        dm, rm = np.broadcast_to(d, out.shape)[meio], rr[meio]
        a1 = np.arccos(np.clip((dm ** 2 + rm ** 2 - R ** 2) / (2 * dm * rm), -1, 1))
        a2 = np.arccos(np.clip((dm ** 2 + R ** 2 - rm ** 2) / (2 * dm * R), -1, 1))
        area = (rm ** 2 * (a1 - np.sin(2 * a1) / 2)
                + R ** 2 * (a2 - np.sin(2 * a2) / 2))
        out[meio] = area
    return out


def fracao_no_quadrado(dx, dy, rc, meio_lado, n=400, rng=None):
    """Fraction of the crown disc inside a square, by Monte Carlo.

    There is no cheap closed form for a disc against a square when the disc crosses a corner, and
    the fraction is what matters, so sampling is enough. n = 400 gives a standard error of 2.5
    points per tree, and what is reported is a mean over hundreds of trees.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    t = rng.random((n, 2))
    ang = 2 * np.pi * t[:, 0]
    rad = np.sqrt(t[:, 1])
    ux, uy = rad * np.cos(ang), rad * np.sin(ang)
    px = dx[:, None] + rc[:, None] * ux[None, :]
    py = dy[:, None] + rc[:, None] * uy[None, :]
    return ((np.abs(px) <= meio_lado) & (np.abs(py) <= meio_lado)).mean(1)


def referencias(cx, cy, Rp, X, Y, Rc, basal):
    """Discrete and continuous reference for a circle of radius Rp centred on (cx, cy)."""
    d = np.hypot(X - cx, Y - cy)
    perto = d <= (Rp + Rc.max())
    dentro = d <= Rp
    w = np.zeros(len(X))
    w[perto] = intersecao(d[perto], Rc[perto], Rp) / (np.pi * Rc[perto] ** 2)
    A_ha = np.pi * Rp ** 2 / 1e4
    return dict(N_disc=dentro.sum() / A_ha, N_cont=w.sum() / A_ha,
                G_disc=basal[dentro].sum() / A_ha, G_cont=(w * basal).sum() / A_ha)


def grade_de_centros(X, Y, Rp, folga):
    x0, x1 = X.min() + Rp + folga, X.max() - Rp - folga
    y0, y1 = Y.min() + Rp + folga, Y.max() - Rp - folga
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 2))
    gx = np.arange(x0, x1 + 1e-9, PASSO_GRADE)
    gy = np.arange(y0, y1 + 1e-9, PASSO_GRADE)
    return np.array([(a, b) for a in gx for b in gy])


def geometria(X, Y, dap_cm, basal, a_copa, escala):
    """Parts 1 and 2: the discrepancy between the two references, by plot size."""
    Rc = a_copa * escala * dap_cm
    linhas = []
    for A in AREAS:
        Rp = np.sqrt(A / np.pi)
        C = grade_de_centros(X, Y, Rp, Rc.max())
        if len(C) == 0:
            continue
        r = pd.DataFrame([referencias(cx, cy, Rp, X, Y, Rc, basal) for cx, cy in C])
        for alvo in ("N", "G"):
            dsc, cnt = r[f"{alvo}_disc"].values, r[f"{alvo}_cont"].values
            dif = cnt - dsc
            rel = 100 * np.sqrt(np.mean(dif ** 2)) / dsc.mean()
            linhas.append(dict(area_m2=A, raio_m=Rp, alvo=alvo, escala_copa=escala,
                               raio_copa_medio=float(Rc.mean()), n_parcelas=len(C),
                               media_disc=float(dsc.mean()), media_cont=float(cnt.mean()),
                               dp_disc=float(dsc.std(ddof=1)), dp_cont=float(cnt.std(ddof=1)),
                               vies_rel=100 * float(dif.mean()) / dsc.mean(),
                               desacordo_rel=float(rel),
                               r_entre_refs=float(np.corrcoef(dsc, cnt)[0, 1])))
    return pd.DataFrame(linhas)


def proveniencia_do_dossel(X, Y, Rc, rng):
    """Part 3: how much of the canopy over the evaluated area belongs to trees OUTSIDE it.

    Asking whether most of the crown falls inside is degenerate: for a convex region that is true
    whenever the stem is inside, by symmetry of the disc, and the ceiling comes out at F1 1.000 on
    both extents. What discriminates is measuring the crown mass, the part that leaves from inside
    and the part that enters from outside, which is the quantity the sensor sees and the discrete
    reference does not count.
    """
    linhas = []
    for nome, meio, Rp in (("26 m square", LADO_QUADRADO / 2, None),
                           ("400 m2 circle", None, RAIO_PARCELA)):
        cx, cy = X.mean(), Y.mean()
        dx, dy = X - cx, Y - cy
        if Rp is None:
            dentro = (np.abs(dx) <= meio) & (np.abs(dy) <= meio)
            perto = (np.abs(dx) <= meio + Rc.max()) & (np.abs(dy) <= meio + Rc.max())
            frac = np.zeros(len(X))
            frac[perto] = fracao_no_quadrado(dx[perto], dy[perto], Rc[perto], meio, rng=rng)
        else:
            d = np.hypot(dx, dy)
            dentro = d <= Rp
            perto = d <= Rp + Rc.max()
            frac = np.zeros(len(X))
            frac[perto] = intersecao(d[perto], Rc[perto], Rp) / (np.pi * Rc[perto] ** 2)
        n_dentro = int(dentro.sum())
        entra = float(frac[~dentro].sum())              # crown of an outside tree that enters
        sai = float((1 - frac[dentro]).sum())           # crown of an inside tree that leaves
        linhas.append(dict(area=nome, n_dentro=n_dentro,
                           copa_que_entra=entra, copa_que_sai=sai,
                           entra_rel=entra / max(n_dentro, 1),
                           sai_rel=sai / max(n_dentro, 1),
                           troca_bruta_rel=(entra + sai) / max(n_dentro, 1),
                           saldo_rel=(entra - sai) / max(n_dentro, 1)))
    return pd.DataFrame(linhas)


def modelo(X, Y, dap_cm, basal, Rc, xyz):
    """Part 4: does the LiDAR predict the continuous reference better? With a smoothing control."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    xy, z = xyz
    linhas = []
    Rc_fixo = np.full(len(X), Rc.mean())
    for A in AREAS:
        Rp = np.sqrt(A / np.pi)
        C = grade_de_centros(X, Y, Rp, Rc.max())
        if len(C) == 0:
            continue
        # spacing of one diameter: the plots hardly touch each other
        passo = max(1, int(round(Rp / PASSO_GRADE)))
        gx = np.unique(C[:, 0])[::passo]
        gy = np.unique(C[:, 1])[::passo]
        C = np.array([(a, b) for a in gx for b in gy])
        if len(C) < 10:
            continue
        F, alvos = [], []
        for cx, cy in C:
            d2 = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
            sel = (d2 <= Rp ** 2) & (z >= config.Z_MIN_M) & (z <= config.Z_MAX_M)
            F.append(stdmetrics_subset(z[sel]))
            r = referencias(cx, cy, Rp, X, Y, Rc, basal)
            r_fix = referencias(cx, cy, Rp, X, Y, Rc_fixo, basal)
            r["N_fixo"], r["G_fixo"] = r_fix["N_cont"], r_fix["G_cont"]
            alvos.append(r)
        F = pd.DataFrame(F)
        T = pd.DataFrame(alvos)
        ok = F.notna().all(1).values
        F, T, C = F[ok], T[ok], C[ok]
        if len(F) < 12:
            continue
        # spatial blocks in x, with a one-radius guard so as not to train on the adjacent neighbour
        blocos = np.digitize(C[:, 0], np.quantile(C[:, 0], [0.25, 0.5, 0.75]))
        for col in ("N_disc", "N_cont", "N_fixo", "G_disc", "G_cont", "G_fixo"):
            y = T[col].values
            pred = np.full(len(y), np.nan)
            for b in np.unique(blocos):
                te = blocos == b
                guarda = np.zeros(len(y), bool)
                for cx, cy in C[te]:
                    guarda |= np.hypot(C[:, 0] - cx, C[:, 1] - cy) < 2 * Rp
                tr = ~te & ~guarda
                if tr.sum() < 6:
                    continue
                m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
                m.fit(F[tr], y[tr])
                pred[te] = m.predict(F[te])
            v = np.isfinite(pred)
            if v.sum() < 6:
                continue
            res = pred[v] - y[v]
            linhas.append(dict(area_m2=A, alvo=col, n=int(v.sum()),
                               media=float(y[v].mean()), dp=float(y[v].std(ddof=1)),
                               rmse=float(np.sqrt(np.mean(res ** 2))),
                               rrmse=100 * float(np.sqrt(np.mean(res ** 2))) / y[v].mean(),
                               r2=1 - np.sum(res ** 2) / np.sum((y[v] - y[v].mean()) ** 2)))
    return pd.DataFrame(linhas)


def main():
    rng = np.random.default_rng(SEED)
    f = gpd.read_file(MAPA)
    X = f.geometry.x.values
    Y = f.geometry.y.values
    dap = pd.read_csv(REF_DAP)
    d_cm = np.where(dap.motivo.values == "ok", dap.dap_cm.values, np.nan)
    n_imp = int(np.isnan(d_cm).sum())
    d_cm = np.where(np.isnan(d_cm), np.nanmedian(d_cm), d_cm)
    basal = np.pi * (d_cm / 200.0) ** 2
    print(f"{len(X)} stems over {np.ptp(X):.0f} x {np.ptp(Y):.0f} m, "
          f"{n_imp} without a measured DBH filled with the median {np.nanmedian(d_cm):.1f} cm")
    print(f"implied basal area {basal.sum() / (np.ptp(X) * np.ptp(Y) / 1e4):.1f} m2/ha\n")

    # ---- 1 and 2. pure geometry, with no model at all -----------------------------------
    print("[1] disagreement between the discrete and the continuous reference, by plot size")
    geos = []
    for esc in VARRE_COPA:
        g = geometria(X, Y, d_cm, basal, A_COPA, esc)
        geos.append(g)
        if esc == 1.0:
            for _, r in g[g.alvo == "G"].iterrows():
                print(f"   {r.area_m2:6.0f} m2  radius {r.raio_m:5.2f} m  n {r.n_parcelas:5.0f}  "
                      f"disagreement {r.desacordo_rel:5.2f} %  bias {r.vies_rel:+5.2f} %  "
                      f"r {r.r_entre_refs:.3f}", flush=True)
    geo = pd.concat(geos, ignore_index=True)

    print("\n   sensitivity to the crown radius, target G, 400 m2 plot")
    for _, r in geo[(geo.alvo == "G") & (geo.area_m2 == 400)].iterrows():
        print(f"   mean crown {r.raio_copa_medio:.2f} m  ->  disagreement {r.desacordo_rel:.2f} %")

    # ---- 3. the ceiling the edge imposes on the count -----------------------------------
    print("\n[2] provenance of the canopy over the evaluated area")
    Rc = A_COPA * d_cm
    teto = proveniencia_do_dossel(X, Y, Rc, rng)
    for _, r in teto.iterrows():
        print(f"   {r.area:16s} n {r.n_dentro:4.0f}  enters from outside {100 * r.entra_rel:5.2f} %  "
              f"leaves from inside {100 * r.sai_rel:5.2f} %  gross exchange {100 * r.troca_bruta_rel:5.2f} %  "
              f"net {100 * r.saldo_rel:+5.2f} %")

    # ---- 4. the model --------------------------------------------------------------------
    print("\n[3] does the LiDAR predict the continuous reference better?")
    las = laspy.read(str(DRONE_T001))
    cls = np.asarray(las.classification)
    m = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    xy = np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m]])
    z = np.asarray(las.z)[m]
    mod = modelo(X, Y, d_cm, basal, Rc, (xy, z))
    for A in sorted(mod.area_m2.unique()):
        s = mod[mod.area_m2 == A].set_index("alvo")
        for alvo in ("N", "G"):
            if f"{alvo}_disc" not in s.index or f"{alvo}_cont" not in s.index:
                continue
            a, b = s.loc[f"{alvo}_disc"], s.loc[f"{alvo}_cont"]
            c = s.loc[f"{alvo}_fixo"] if f"{alvo}_fixo" in s.index else None
            ganho = 100 * (a.rrmse - b.rrmse) / a.rrmse
            extra = f"  fixed kernel {c.rrmse:5.2f} %" if c is not None else ""
            print(f"   {A:6.0f} m2  {alvo}  discrete {a.rrmse:5.2f} %  continuous {b.rrmse:5.2f} %"
                  f"  gain {ganho:+5.1f} %  R2 {a.r2:+.2f} -> {b.r2:+.2f}{extra}")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    geo.to_csv(SAIDA, index=False)
    pd.concat([mod, teto.assign(area_m2=np.nan)], ignore_index=True).to_csv(SAIDA_MOD, index=False)
    print(f"\n{SAIDA}\n{SAIDA_MOD}")


if __name__ == "__main__":
    main()
