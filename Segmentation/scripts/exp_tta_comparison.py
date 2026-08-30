"""Count and matched metric for the two segmenters under test-time adaptation.

Both metrics come out of the same code here, for both models, under any adaptation
condition.

What it measures, per plot of stand 001:

  count      merged instances inside the plot circle, against the field census. It
             reports the 12 m radius the project has used from the start and the
             11.28 m radius that closes the 400 m² the field measured, because the
             12 m one inflates the area by 13% and therefore inflates the count.

  matched    optimal one-to-one assignment against the map of 892 stems, in the
             central 26 m square of the tile. It says whether the model found
             those trees, and separates omission from commission, which the count
             ratio does not do.

Inputs. Each condition is a folder:
  SegmentAnyTree  folder with `t001_pXXX_gY_out.laz`, field `PredInstance`
  FF3D            folder with the panoptic PLYs `*_round2.ply`, or a CSV of
                  already extracted centroids

Run:
    PYTHONPATH=. python scripts/exp_tta_comparison.py \
        --sat baseline=<dir> --sat adabn=<dir> \
        --ff3d baseline=<csv|dir> --ff3d adabn=<dir> \
        --out manual_match/tta_comparison.csv
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ff3d_detection_numbers import find_plys  # noqa: E402
from ff3d_detection_overlap import instance_centroids, nms_merge  # noqa: E402

from greenvista import config  # noqa: E402

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
BACKUP = Path(config.LAZ_DIR).parent / "ff3d_tiles_overlap"
PLOTS = ["1_1", "1_2"]
RAIO_FUSAO = 1.5            # same merge radius for both models
RAIO_12 = 12.0              # the 12 m radius the project also reports, 452 m²
RAIO_400 = float(np.sqrt(400 / np.pi))   # 11.28 m, the 400 m² the field measured
TILE, MARGEM = 32.0, 3.0    # central 26 m square for the matched metric
LIMIARES = [1.0, 1.5, 2.0, 2.5, 3.0]
FAIXA_BASE = 3.0            # height of the band used to estimate the stem base


def posicao_da_instancia(xy, z, solo, modo):
    """Where the instance "is", by the crown centroid or by the stem base.

    The crown centroid is what the project had been using, because it is what FF3D
    delivers. It sits metres away from the stem when the tree leans or the crown is
    asymmetric, and that penalises matching against a map of STEMS without the
    model having detected any worse. Separating the two is exactly the point of
    this function.

    The base is estimated by the same rule in both models: throw away whatever each
    one called ground, take the points in the first few metres above the lowest
    remaining point, and average. If too few points remain, it falls back to the
    centroid, because a base estimated from half a dozen points is worse than none.
    """
    if modo == "centroide":
        return xy.mean(0)
    m = ~solo if solo is not None else np.ones(len(z), bool)
    if m.sum() < 20:
        return xy.mean(0)
    zb = z[m]
    faixa = m & (z <= zb.min() + FAIXA_BASE)
    if faixa.sum() < 20:
        return xy.mean(0)
    return xy[faixa].mean(0)


# ---------------------------------------------------------------- references
def centros_e_censo():
    p = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    inv = pd.read_csv(config.DATA / "5-dados_campo/inventario_est.csv")
    p = p.dropna(subset=["talhao", "parcela"]).merge(
        inv[["talhao", "parcela", "n_arv"]], on=["talhao", "parcela"], how="left")
    return {f"{int(r.talhao)}_{int(r.parcela)}":
            (r.geometry.x, r.geometry.y, r.n_arv) for _, r in p.iterrows()}


# ------------------------------------------------------------------- inputs
def centroides_sat(pasta, modo="centroide"):
    """One position per predicted instance, grouped by plot.

    The SegmentAnyTree outputs store the coordinate in UTM, so there is no shift to
    undo, and z already comes normalised by the terrain height. The weight used in
    the NMS is the instance point count, the same as the FF3D side uses, so that
    the merge is the same rule in both.
    """
    pasta = Path(pasta)
    por_parcela = {}
    for f in sorted(pasta.glob("t001_p*_out.laz")):
        pid = f"1_{int(f.name.split('_p')[1][:3])}"
        las = laspy.read(str(f))
        inst = np.asarray(las.PredInstance)
        xy = np.column_stack([np.asarray(las.x), np.asarray(las.y)])
        z = np.asarray(las.z)
        sem = np.asarray(las.PredSemantic)
        cs, ss = [], []
        for i in np.unique(inst[inst > 0]):
            m = inst == i
            cs.append(posicao_da_instancia(xy[m], z[m], sem[m] == 0, modo))
            ss.append(int(m.sum()))
        if cs:
            por_parcela.setdefault(pid, [[], []])
            por_parcela[pid][0].extend(cs)
            por_parcela[pid][1].extend(ss)
    return {k: (np.asarray(v[0]).reshape(-1, 2), np.asarray(v[1]))
            for k, v in por_parcela.items()}


def posicoes_ff3d_ply(ply, backup_laz, modo):
    """Positions of the instances of a panoptic PLY, in UTM, in the requested mode.

    The shift to UTM is the same one `instance_centroids` uses, the mean of the
    backup tile coordinates, because that is how the pipeline converter took the
    tile into local coordinates. Using a different shift here would put the stem
    base and the centroid in different frames, and the comparison between the two
    modes would lose its meaning.

    The FF3D semantics are 0 ground, 1 wood, 2 leaf.
    """
    from plyfile import PlyData

    el = PlyData.read(str(ply)).elements[0]
    d = {p.name: np.asarray(el[p.name]) for p in el.properties}
    inst = d.get("instance_pred", d.get("instance"))
    sem = d.get("semantic_pred")
    xy = np.column_stack([d["x"], d["y"]])
    z = d["z"]
    las = laspy.read(str(backup_laz))
    mx, my = float(np.asarray(las.x).mean()), float(np.asarray(las.y).mean())
    cents, sizes = [], []
    for iid in np.unique(inst):
        if iid < 0:
            continue
        m = inst == iid
        solo = (sem[m] == 0) if sem is not None else None
        p = posicao_da_instancia(xy[m], z[m], solo, modo)
        cents.append([p[0] + mx, p[1] + my])
        sizes.append(int(m.sum()))
    return np.asarray(cents).reshape(-1, 2), np.asarray(sizes)


def centroides_ff3d(origem, modo="centroide"):
    """Same for FF3D, accepting either the PLYs or an already extracted CSV.

    The CSV stores only the centroid, so there is no way to ask it for a stem base.
    In that case the script refuses, rather than returning a centroid in place of
    the base.
    """
    origem = Path(origem)
    if origem.is_file() and origem.suffix == ".csv":
        if modo != "centroide":
            raise SystemExit(f"{origem.name} only stores centroids; for a stem "
                             f"base point at the folder of PLYs")
        c = pd.read_csv(origem)
        c = c[c.plot_id.isin(PLOTS)]
        return {pid: (g[["cx", "cy"]].to_numpy(float), g["n_pts"].to_numpy(float))
                for pid, g in c.groupby("plot_id")}
    por_parcela = {}
    for ply in find_plys(str(origem)):
        nome = Path(ply).name.split("_round2")[0].replace("fixedname", "")
        tile = nome if (BACKUP / f"{nome}.laz").exists() else None
        if tile is None:                       # the pipeline sanitises the name
            cand = [p.stem for p in BACKUP.glob("t001_*.laz") if nome.startswith(p.stem)]
            if not cand:
                print(f"  no backup tile for {Path(ply).name}, skipped")
                continue
            tile = cand[0]
        pid = f"1_{int(tile.split('_p')[1][:3])}"
        if modo == "centroide":
            cents, sizes, _ = instance_centroids(ply, BACKUP / f"{tile}.laz")
        else:
            cents, sizes = posicoes_ff3d_ply(ply, BACKUP / f"{tile}.laz", modo)
        por_parcela.setdefault(pid, [[], []])
        por_parcela[pid][0].extend(cents.tolist())
        por_parcela[pid][1].extend(sizes.tolist())
    return {k: (np.asarray(v[0]).reshape(-1, 2), np.asarray(v[1]))
            for k, v in por_parcela.items()}


# -------------------------------------------------------------------- metrics
def no_quadrado(xy, cx, cy, meia=TILE / 2 - MARGEM):
    if len(xy) == 0:
        return xy
    return xy[(np.abs(xy[:, 0] - cx) <= meia) & (np.abs(xy[:, 1] - cy) <= meia)]


def parear(ref, pred, limiar):
    """Optimal one-to-one assignment, not greedy nearest neighbour.

    The greedy version lets two predictions land on the same tree and therefore
    overstates accuracy. Returns the number of valid pairs and their position RMSE.
    """
    if len(ref) == 0 or len(pred) == 0:
        return 0, float("nan")
    d = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    li, ci = linear_sum_assignment(np.where(d <= limiar, d, 1e6))
    ok = d[li, ci] <= limiar
    rmse = float(np.sqrt((d[li, ci][ok] ** 2).mean())) if ok.any() else float("nan")
    return int(ok.sum()), rmse


def avalia(nome_modelo, nome_cond, dados, ctr, REF):
    linhas_c, linhas_m = [], []
    for pid in PLOTS:
        if pid not in dados:
            print(f"  {nome_modelo}/{nome_cond}: plot {pid} has no output, skipped")
            continue
        cents, sizes = dados[pid]
        cx, cy, campo = ctr[pid]
        fundidos = nms_merge(cents, sizes, RAIO_FUSAO)

        d = np.hypot(fundidos[:, 0] - cx, fundidos[:, 1] - cy) if len(fundidos) else np.array([])
        n12, n400 = int((d <= RAIO_12).sum()), int((d <= RAIO_400).sum())
        linhas_c.append(dict(modelo=nome_modelo, condicao=nome_cond, parcela=pid,
                             campo=int(campo), agrupadas=len(cents), fundidas=len(fundidos),
                             n_r12=n12, det_r12=n12 / campo,
                             n_r400=n400, det_r400=n400 / campo))

        ref = no_quadrado(REF, cx, cy)
        pred = no_quadrado(fundidos, cx, cy)
        for lim in LIMIARES:
            tp, rmse = parear(ref, pred, lim)
            prec = tp / len(pred) if len(pred) else 0.0
            rec = tp / len(ref) if len(ref) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            linhas_m.append(dict(modelo=nome_modelo, condicao=nome_cond, parcela=pid,
                                 limiar_m=lim, ref=len(ref), pred=len(pred), TP=tp,
                                 FP=len(pred) - tp, FN=len(ref) - tp,
                                 precisao=prec, revocacao=rec, F1=f1, rmse_pos_m=rmse))
    return linhas_c, linhas_m


def par(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected condition=path, got {s}")
    k, v = s.split("=", 1)
    return k, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sat", type=par, action="append", default=[])
    ap.add_argument("--ff3d", type=par, action="append", default=[])
    ap.add_argument("--out", type=Path,
                    default=config.OUT_DIR / "tta_comparison.csv")
    ap.add_argument("--posicao", choices=["centroide", "base"], default="centroide",
                    help="instance centroid, or stem base estimated in the first "
                         "few metres above the ground")
    args = ap.parse_args()

    ctr = centros_e_censo()
    ref_todo = gpd.read_file(MAPA)
    REF = np.column_stack([ref_todo.geometry.x, ref_todo.geometry.y])
    print(f"reference: {len(REF)} stems in the map of stand 001")
    print(f"view merging: NMS at {RAIO_FUSAO} m, identical for both models")
    print(f"predicted position: {args.posicao}\n")

    cont, casa = [], []
    for modelo, itens, carregador in [("SegmentAnyTree", args.sat, centroides_sat),
                                      ("FF3D", args.ff3d, centroides_ff3d)]:
        for cond, caminho in itens:
            print(f"reading {modelo} / {cond}: {caminho}")
            dados = carregador(caminho, args.posicao)
            if not dados:
                print(f"  NOTHING loaded from {caminho} — condition ignored")
                continue
            c, m = avalia(modelo, cond, dados, ctr, REF)
            cont += c
            casa += m

    if not cont:
        raise SystemExit("no condition produced a result, nothing to write")

    dc, dm = pd.DataFrame(cont), pd.DataFrame(casa)

    print("\n=== count inside the plot ===")
    print(f"{'model':>15} {'cond':>10} {'plot':>5} {'field':>6} {'merg':>6} "
          f"{'r12':>5} {'det12':>7} {'r400':>5} {'det400':>7}")
    for _, r in dc.iterrows():
        print(f"{r.modelo:>15} {r.condicao:>10} {r.parcela:>5} {r.campo:>6} {r.fundidas:>6} "
              f"{r.n_r12:>5} {r.det_r12:>6.1%} {r.n_r400:>5} {r.det_r400:>6.1%}")

    print("\n=== matched metric, the two plots summed ===")
    tot = dm.groupby(["modelo", "condicao", "limiar_m"])[["ref", "pred", "TP", "FP", "FN"]].sum().reset_index()
    tot["precisao"] = tot.TP / tot.pred.replace(0, np.nan)
    tot["revocacao"] = tot.TP / tot.ref
    tot["F1"] = 2 * tot.precisao * tot.revocacao / (tot.precisao + tot.revocacao)
    print(f"{'model':>15} {'cond':>10} {'lim':>5} {'ref':>5} {'pred':>5} {'TP':>4} "
          f"{'FP':>4} {'FN':>4} {'prec':>7} {'recall':>7} {'F1':>7}")
    for _, r in tot.sort_values(["modelo", "condicao", "limiar_m"]).iterrows():
        print(f"{r.modelo:>15} {r.condicao:>10} {r.limiar_m:>5.1f} {int(r.ref):>5} "
              f"{int(r.pred):>5} {int(r.TP):>4} {int(r.FP):>4} {int(r.FN):>4} "
              f"{r.precisao:>6.1%} {r.revocacao:>6.1%} {r.F1:>6.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dc.to_csv(args.out.with_name(args.out.stem + "_contagem.csv"), index=False)
    dm.to_csv(args.out.with_name(args.out.stem + "_casada.csv"), index=False)
    tot.to_csv(args.out.with_name(args.out.stem + "_casada_total.csv"), index=False)
    print(f"\nwritten to {args.out.parent}")


if __name__ == "__main__":
    main()
