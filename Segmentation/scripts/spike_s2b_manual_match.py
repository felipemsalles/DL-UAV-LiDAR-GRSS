#!/usr/bin/env python3
"""Spike S2b (manual-match variant) — Tier-A feasibility test on FF3D crowns.

Unlike spike_s2b_crown_regression.py (which assigns trees by an automated low-band stem vote),
this uses the *hand-verified* arv->instance matches read off CloudCompare by point-picking the
crown apex above each cubed-tree stem (2026-05-23). 7 of 9 geolocated trees matched cleanly;
arv 4 and 18 had a crown at the right height but FF3D left it unlabeled (instance_pred = -1) ->
genuinely unmatched, a segmentation-quality result.

Extract per-instance crown features and regress crown features -> V (LOOCV + bootstrap CI).
Compare to: crude-cylinder S1 (R²=0.19) and the field-height ceiling (R²=0.78). N=7 -> directional,
very wide CIs; all feature combos are reported (not cherry-picked) given the leakage risk at this N.
"""
import numpy as np, pandas as pd
from plyfile import PlyData
from greenvista.features import crown_features
from greenvista.volume import CrownAllometry
from greenvista.eval import loocv_predict, metrics, bootstrap_ci

from greenvista import config
DATA = config.DATA
PLY = str(config.FF3D_TILE_PLY)

# Hand-verified matches (CloudCompare point-picking, 2026-05-23). arv -> FF3D instance_pred.
MANUAL = {5: 159, 19: 65, 20: 87, 21: 1, 22: 58, 23: 139, 24: 120}
UNMATCHED = [4, 18]  # crown present at field height but instance_pred == -1 (FF3D miss)


def main():
    # --- ground truth: V and field height per destructively scaled tree (stand 001) ---
    g = pd.read_csv(DATA/"2-shapes/Cubagem_T001/arvores_cubadas.csv")
    g = g[g.Name.astype(str).str.fullmatch(r"\d+")].copy(); g["arv"] = g.Name.astype(int)
    vol = pd.read_csv(DATA/"5-dados_campo/volumes.csv")
    lk = g.merge(vol[vol.talhao == 1][["arv", "V", "H_m"]], on="arv").set_index("arv")

    # --- FF3D instance cloud (local coords are fine: crown features are translation-invariant) ---
    el = PlyData.read(PLY).elements[0]
    d = {p.name: np.array(el[p.name]) for p in el.properties}
    xyz = np.column_stack([d['x'], d['y'], d['z']]); inst = d['instance_pred']

    # --- per-instance crown features for the matched trees ---
    rows = []
    for arv, iid in MANUAL.items():
        pts = xyz[inst == iid]
        f = crown_features(pts)
        f.update(arv=arv, inst=iid, V=float(lk.loc[arv, "V"]), field_H=float(lk.loc[arv, "H_m"]))
        rows.append(f)
    df = pd.DataFrame(rows).sort_values("arv").reset_index(drop=True)

    show = ["arv", "inst", "V", "field_H", "n_points", "H_max", "z99",
            "crown_area", "crown_volume", "crown_diameter"]
    show = [c for c in show if c in df.columns]
    print(f"Matched N={len(df)} (unmatched: {UNMATCHED})\n")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(df[show].to_string(index=False))

    # --- height fidelity & feature/V correlations ---
    print("\ncorr(FF3D H_max, field_H) = %+.2f  (bias %+.2f m)"
          % (np.corrcoef(df.H_max, df.field_H)[0, 1], (df.H_max - df.field_H).mean()))
    for c in ["H_max", "z99", "crown_area", "crown_volume", "crown_diameter", "n_points"]:
        if c in df.columns and df[c].notna().all():
            print(f"  r(V, {c:14s}) = {np.corrcoef(df.V, df[c])[0,1]:+.2f}")

    # --- LOOCV crown-allometry regressions (every combination, no selection) ---
    print("\nLOOCV  V ~ crown features  (CrownAllometry):")
    combos = [["crown_area", "H_max"], ["crown_volume", "H_max"], ["crown_area"],
              ["crown_volume"], ["crown_area", "crown_volume", "H_max"]]
    for cols in combos:
        if not all(c in df.columns for c in cols):
            continue
        sub = df.dropna(subset=cols)
        if len(sub) >= 5 and (sub[cols].values > 0).all():
            yhat = loocv_predict(CrownAllometry, sub[cols].values, sub.V.values)
            mm = metrics(sub.V.values, yhat); ci = bootstrap_ci(sub.V.values, yhat)
            print(f"  V ~ {'+'.join(cols):32s} N={len(sub)}  R²={mm['R2']:+.2f} "
                  f"(CI {ci['R2_ci'][0]:+.2f}..{ci['R2_ci'][1]:+.2f})  rRMSE={mm['rRMSE_pct']:.0f}%")

    # --- reference: V from FF3D crown H_max alone (the cheapest possible predictor) ---
    yhat = loocv_predict(CrownAllometry, df[["H_max"]].values, df.V.values)
    mm = metrics(df.V.values, yhat); ci = bootstrap_ci(df.V.values, yhat)
    print(f"  V ~ {'H_max (FF3D)':32s} N={len(df)}  R²={mm['R2']:+.2f} "
          f"(CI {ci['R2_ci'][0]:+.2f}..{ci['R2_ci'][1]:+.2f})  rRMSE={mm['rRMSE_pct']:.0f}%")

    print("\nReference: crude-cylinder S1 R²=0.19 ; field-H ceiling R²=0.78. "
          "N=7 -> directional only, CIs are wide.")


if __name__ == "__main__":
    main()
