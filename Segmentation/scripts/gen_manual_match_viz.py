#!/usr/bin/env python3
"""Generate manual tree<->instance matching aids for the FF3D tile (stand 001).

Outputs to manual_match/:
  - per_tree_arvNN.png : top-down + side view of each cubed tree's neighbourhood, colored by FF3D
    instance, stem starred, candidate instance IDs labeled, field height annotated.
  - candidates.csv     : numeric shortlist per tree (candidate instances + diagnostics + auto best guess).
  - cc_cloud.ply       : tile points (ABS UTM) colored by instance — open in CloudCompare.
  - cc_markers.ply     : vertical stem markers at each cubed tree (ABS UTM), scalar `arv`.
"""
import numpy as np, pandas as pd, laspy
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement

from greenvista import config
DATA = config.DATA
PLY = str(config.FF3D_TILE_PLY)
OUT = config.OUT_DIR; OUT.mkdir(parents=True, exist_ok=True)
R_VIEW, R_CAND = 4.0, 2.5

def rgb_for(ids):
    rng = np.random.default_rng(7)
    uniq = np.unique(ids[ids >= 0])
    cmap = {int(u): (rng.integers(40, 255, 3)) for u in uniq}
    out = np.full((len(ids), 3), 130, np.uint8)  # gray for unassigned
    for u, c in cmap.items():
        out[ids == u] = c
    return out

def main():
    g = pd.read_csv(DATA/"2-shapes/Cubagem_T001/arvores_cubadas.csv")
    g = g[g.Name.astype(str).str.fullmatch(r"\d+")].copy(); g["arv"] = g.Name.astype(int)
    vol = pd.read_csv(DATA/"5-dados_campo/volumes.csv")
    lk = g.merge(vol[vol.talhao==1][["arv","V","H_m"]], on="arv")
    E, N = lk.Easting.values, lk.Northing.values
    e0, e1 = 748945, 748980; n0, n1 = N.min()-3, N.max()+3
    trees = lk[(E>=e0)&(E<=e1)&(N>=n0)&(N<=n1)].copy().reset_index(drop=True)

    las = laspy.read(config.TILE_LAZ)
    x, y = np.asarray(las.x), np.asarray(las.y); m = (x>=e0)&(x<=e1)&(y>=n0)&(y<=n1)
    ax0, ay0 = x[m].min(), y[m].min()
    el = PlyData.read(PLY).elements[0]; d = {p.name: np.array(el[p.name]) for p in el.properties}
    sx, sy = ax0 - d['x'].min(), ay0 - d['y'].min()
    X, Y, Z, inst = d['x']+sx, d['y']+sy, d['z'], d['instance_pred']
    rgb = rgb_for(inst)

    # CloudCompare files (ABS UTM)
    cloud = np.empty(len(X), dtype=[("x","f4"),("y","f4"),("z","f4"),("instance_pred","i4"),
                                    ("red","u1"),("green","u1"),("blue","u1")])
    cloud["x"],cloud["y"],cloud["z"],cloud["instance_pred"] = X,Y,Z,inst
    cloud["red"],cloud["green"],cloud["blue"] = rgb[:,0],rgb[:,1],rgb[:,2]
    PlyData([PlyElement.describe(cloud,"vertex")]).write(str(OUT/"cc_cloud.ply"))
    mk = []
    for _,t in trees.iterrows():
        for zz in np.linspace(0, t.H_m, 60):
            mk.append((t.Easting, t.Northing, zz, int(t.arv)))
    mka = np.array(mk, dtype=[("x","f4"),("y","f4"),("z","f4"),("arv","i4")])
    PlyData([PlyElement.describe(mka,"vertex")]).write(str(OUT/"cc_markers.ply"))

    # per-tree panels + candidate shortlist
    rows = []
    for _,t in trees.iterrows():
        near = (X-t.Easting)**2+(Y-t.Northing)**2 <= R_VIEW**2
        xs, ys, zs, isn = X[near], Y[near], Z[near], inst[near]
        cand = (X-t.Easting)**2+(Y-t.Northing)**2 <= R_CAND**2
        ci = inst[cand]; ci = ci[ci>=0]
        fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
        axs[0].scatter(xs, ys, c=rgb[near]/255, s=4)
        axs[0].plot(t.Easting, t.Northing, "k*", ms=18, mec="w")
        axs[0].set(title=f"tree {int(t.arv)}  field H={t.H_m:.1f}m  V={t.V:.3f}m³  (top-down)",
                   xlabel="E", ylabel="N", aspect="equal")
        axs[1].scatter(xs, zs, c=rgb[near]/255, s=4)
        axs[1].axvline(t.Easting, color="k", ls="--", lw=1)
        axs[1].axhline(t.H_m, color="r", ls=":", lw=1, label=f"field H={t.H_m:.1f}")
        axs[1].set(title="side (E-Z)", xlabel="E", ylabel="Z (m)"); axs[1].legend(fontsize=8)
        # label candidate instance ids at their local centroids
        best, bestn = -1, 0
        for u in np.unique(ci):
            mm = isn == u; n_in = int((ci==u).sum())
            if n_in > bestn: best, bestn = int(u), n_in
            cx, cy, ch = xs[mm].mean(), ys[mm].mean(), zs[mm].max()
            axs[0].annotate(str(int(u)), (cx, cy), fontsize=9, weight="bold")
            axs[1].annotate(str(int(u)), (cx, ch), fontsize=9, weight="bold")
            rows.append({"arv": int(t.arv), "field_H": t.H_m, "V": t.V, "inst": int(u),
                         "n_pts_in_2.5m": n_in, "inst_total_pts": int((inst==u).sum()),
                         "inst_Hmax": float(Z[inst==u].max()),
                         "centroid_dist_m": float(np.hypot(X[inst==u].mean()-t.Easting, Y[inst==u].mean()-t.Northing))})
        fig.suptitle(f"tree {int(t.arv)} — candidate instances (auto-guess by point-vote: {best})", fontsize=11)
        fig.tight_layout(); fig.savefig(OUT/f"per_tree_arv{int(t.arv):02d}.png", dpi=110); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/"candidates.csv", index=False)
    print("wrote", OUT, "->", sorted(p.name for p in OUT.iterdir()))
    print("\nCloudCompare: copy cc_cloud.ply (+cc_markers.ply) to Windows; color cloud by 'instance_pred' "
          "(or RGB); markers are vertical stems colored by 'arv'.")

if __name__ == "__main__":
    main()
