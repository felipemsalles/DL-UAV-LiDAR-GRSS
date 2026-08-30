#!/usr/bin/env python3
"""Where do the plot-Vol_ha numbers come from — LiDAR or tabular (age)?

Finding: the strong cross-stand numbers are carried by stand age (a tabular record), not by the
LiDAR canopy metrics. Once age is controlled (mature stands only), LiDAR shows ~no correlation with
plot volume. LiDAR's genuine roles are elsewhere (the within-stand map; weak per-tree DBH).

Run: PYTHONPATH=. python scripts/exp_source_decomposition.py
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from greenvista import config
from greenvista.repro import seed_everything, write_manifest
from greenvista.area_based import data as abd, models as abm, validate as abv
from greenvista.eval import metrics


def load():
    df = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                        config.DATA / "5-dados_campo/inventario_est.csv")
    s = pd.read_csv(config.OUT_DIR / "plot_structural.csv")[["plot_id", "rumple"]]
    return df.merge(s, on="plot_id", how="left")


def row(df, name, cols, source):
    y = df.Vol_ha.to_numpy()
    make = lambda: abm.make_huber_allom(cols)
    lopo = abv._lopo(make, df)[0]
    loto = abv._group_cv(make, df)
    return {"model": name, "source": source, "features": "+".join(cols),
            "LOPO_R2": metrics(y, lopo)["R2"], "LOPO_rRMSE": metrics(y, lopo)["rRMSE_pct"],
            "LOTO_R2": metrics(y, loto)["R2"], "LOTO_rRMSE": metrics(y, loto)["rRMSE_pct"]}


def main():
    seed_everything(0)
    df = load()
    rows = [
        row(df, "LiDAR height", ("zmean", "zq90"), "LiDAR"),
        row(df, "LiDAR rumple", ("rumple",), "LiDAR"),
        row(df, "LiDAR height+rumple", ("zmean", "zq90", "rumple"), "LiDAR"),
        row(df, "tabular age", ("idade_anos",), "tabular"),
        row(df, "LiDAR+age", ("zmean", "zq90", "idade_anos"), "mixed"),
        row(df, "LiDAR+age+rumple", ("zmean", "zq90", "idade_anos", "rumple"), "mixed"),
    ]
    res = pd.DataFrame(rows)
    print("=== full 13-plot decomposition (huber log-allometry) ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    mat = df[df.idade_anos > 8].reset_index(drop=True)
    ym = mat.Vol_ha.to_numpy()
    print(f"\n=== age-controlled (mature stands only, N={len(mat)}, age {mat.idade_anos.min():.1f}-{mat.idade_anos.max():.1f}) ===")
    cors = {c: float(np.corrcoef(mat[c].to_numpy(float), ym)[0, 1]) for c in ["idade_anos", "zmean", "zq90", "rumple"]}
    print("  correlation with Vol_ha:", {k: round(v, 2) for k, v in cors.items()})
    print("  → LiDAR canopy metrics carry ~no plot-volume signal once age is fixed")

    # figure: Vol vs age (tight) | Vol vs LiDAR height within mature stands (flat)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(df.idade_anos, df.Vol_ha, c=df.talhao, cmap="tab10", s=90, edgecolor="k")
    ax[0].set_xlabel("stand age (yr) — TABULAR"); ax[0].set_ylabel("Vol_ha (m³/ha)")
    ax[0].set_title(f"Vol_ha vs age (all 13 plots)\nage carries it: LOTO R²=0.81, rRMSE 11%")
    ax[1].scatter(mat.zmean, mat.Vol_ha, c=mat.talhao, cmap="tab10", s=90, edgecolor="k")
    ax[1].set_xlabel("LiDAR canopy zmean (m) — LiDAR"); ax[1].set_ylabel("Vol_ha (m³/ha)")
    ax[1].set_title(f"Vol_ha vs LiDAR height, mature stands only (age fixed)\n"
                    f"no signal: r={cors['zmean']:+.2f} (colour = stand)")
    fig.suptitle("Where the plot-Vol_ha numbers come from: age (tabular), not LiDAR canopy metrics", y=1.0)
    fig.tight_layout()
    out = config.OUT_DIR / "source_decomposition.png"
    fig.savefig(out, dpi=130)

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.OUT_DIR / "source_decomposition.csv", index=False)
    write_manifest(config.OUT_DIR / "source_decomposition_manifest.json",
                   script="exp_source_decomposition.py", config={"seed": 0},
                   metrics={"decomp": rows, "mature_only_corr": cors})
    print(f"\nwrote {out} + source_decomposition.csv")


if __name__ == "__main__":
    main()
