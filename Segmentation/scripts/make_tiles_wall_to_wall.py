#!/usr/bin/env python3
"""Grid of tiles covering the whole stands, and not only the surroundings of the plots.

`ff3d_make_tiles_overlap.py` builds only a 3x3 around the centre of one plot,
because it was made to test edge loss at 13 known points. To answer "how many
trees does the stand have" the grid has to sweep the entire polygon, and its
phase is arbitrary: no plot falls at a tile centre.

Geometry. 32 m tiles, because tile size determines the VRAM and 32 m is what fits
in 8 GB. The stride is 32/3 = 10.67 m, which makes every interior point fall in
3x3 = 9 distinct tiles: the same redundancy as the "9 views" of the per-plot
scheme, applied continuously instead of in isolated blocks.

Edge margin. The grid extends 16 m beyond the polygon (an empty tile is cheap and
the point floor discards it), but that buys no context: the clouds were clipped
exactly to the polygon, with zero points outside among 1.8 million sampled across
the six. The crown of the outermost-row tree is cut at the boundary for lack of
data, and no grid arrangement fixes that. What quantifies the effect on the final
count is `exp_w2w_contagem.py`, in the ring column.

Point floor. A tile with fewer than MIN_PTS points is discarded without being
written. In the margin outside the polygon most tiles fall here.

Run:
    PYTHONPATH=. python scripts/make_tiles_wall_to_wall.py --contar
    PYTHONPATH=. python scripts/make_tiles_wall_to_wall.py --out work/tiles_w2w
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from shapely.geometry import box

from greenvista import config

SIZE = 32.0
STRIDE = SIZE / 3.0        # 10.67 m -> 9 tiles per interior point
FOLGA = 16.0               # half a tile width outside the polygon
MIN_PTS = 20_000           # ~20 pts/m² in a 1024 m² tile; below that it is empty edge


def talhoes_com_nuvem(laz_dir: Path):
    """The stands that have a .laz file, read from disk and not from a constant."""
    return sorted(int(f.stem.split("_")[-1]) for f in laz_dir.glob("SaoManuelTotal_*.laz"))


def poligonos():
    a = gpd.read_file(config.DATA / "2-shapes/Areas_plantio/area_plantio.shp")
    a["t"] = pd.to_numeric(a.Talhao, errors="coerce").astype("Int64")
    return {int(r.t): r.geometry for _, r in a.iterrows() if pd.notna(r.t)}


def centros_da_grade(poly):
    """Tile centres on a regular grid covering the polygon plus the margin.

    Keeps the tile whose 32 m square intersects the polygon. A tile whose centre
    is outside but whose area touches the stand still contributes, and it is
    precisely that tile which gives context to the boundary tree.
    """
    x0, y0, x1, y1 = poly.bounds
    xs = np.arange(x0 - FOLGA, x1 + FOLGA + STRIDE, STRIDE)
    ys = np.arange(y0 - FOLGA, y1 + FOLGA + STRIDE, STRIDE)
    h = SIZE / 2.0
    out = []
    for cx in xs:
        for cy in ys:
            if poly.intersects(box(cx - h, cy - h, cx + h, cy + h)):
                out.append((float(cx), float(cy)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laz", type=Path, default=config.REPO / "work/lazall")
    ap.add_argument("--out", type=Path, default=config.REPO / "work/tiles_w2w")
    ap.add_argument("--contar", action="store_true",
                    help="only counts the grid tiles and estimates the cost, does not read the cloud")
    ap.add_argument("--contar-real", action="store_true",
                    help="counts how many tiles SURVIVE the point floor, without writing")
    args = ap.parse_args()

    polys = poligonos()
    ts = talhoes_com_nuvem(args.laz)
    if not ts:
        raise SystemExit(f"no cloud in {args.laz}")
    print(f"stands with a cloud on disk: {ts}\n")

    if args.contar:
        tot = 0
        print(f"{'stand':>7} {'ha':>6} {'tiles':>7} {'predicted':>12}")
        print("-" * 40)
        for t in ts:
            n = len(centros_da_grade(polys[t]))
            tot += n
            print(f"{t:>7} {polys[t].area/1e4:6.2f} {n:>7} {n*49/3600:11.1f}h")
        print("-" * 40)
        print(f"{'TOTAL':>7} {sum(polys[t].area for t in ts)/1e4:6.2f} {tot:>7} "
              f"{tot*49/3600:11.1f}h")
        print("\n(at 49 s/tile, the rate measured in this week\'s runs; the floor of "
              f"{MIN_PTS:,} points will still drop the empty-margin tiles)")
        return

    if args.contar_real:
        # A 32 m tile is exactly 3x3 STRIDE cells, so the point count comes from
        # a 2-D histogram summed over a 3x3 window. Instantaneous, and exact,
        # without sweeping the cloud once per tile.
        tot = surv = 0
        print(f"{'stand':>7} {'grid':>7} {'surviving':>11} {'median pts/tile':>18}")
        print("-" * 48)
        for t in ts:
            las = laspy.read(str(args.laz / f"SaoManuelTotal_{t:03d}.laz"))
            x, y = np.asarray(las.x), np.asarray(las.y)
            centros = centros_da_grade(polys[t])
            cx = np.array([c[0] for c in centros]); cy = np.array([c[1] for c in centros])
            x0, y0 = cx.min() - SIZE, cy.min() - SIZE
            nx = int((cx.max() + SIZE - x0) / STRIDE) + 2
            ny = int((cy.max() + SIZE - y0) / STRIDE) + 2
            H, _, _ = np.histogram2d(x, y, bins=[nx, ny],
                                     range=[[x0, x0 + nx * STRIDE], [y0, y0 + ny * STRIDE]])
            # 2-D cumulative sum, for 3x3 windows in O(1) each
            S = np.pad(H.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            ix = ((cx - SIZE / 2 - x0) / STRIDE).round().astype(int)
            iy = ((cy - SIZE / 2 - y0) / STRIDE).round().astype(int)
            ix = np.clip(ix, 0, nx - 3); iy = np.clip(iy, 0, ny - 3)
            n = (S[ix + 3, iy + 3] - S[ix, iy + 3] - S[ix + 3, iy] + S[ix, iy]).astype(int)
            ok = n >= MIN_PTS
            tot += len(centros); surv += int(ok.sum())
            print(f"{t:>7} {len(centros):>7} {int(ok.sum()):>11} {int(np.median(n[ok])):>18,}")
            del las, x, y, H, S
        print("-" * 48)
        print(f"{'TOTAL':>7} {tot:>7} {surv:>11}")
        print(f"\nprediction at 49 s/tile: {surv*49/3600:.1f} h  "
              f"(the full grid would give {tot*49/3600:.1f} h)")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.out.glob("*.laz"):
        f.unlink()

    linhas = []
    h = SIZE / 2.0
    for t in ts:
        las = laspy.read(str(args.laz / f"SaoManuelTotal_{t:03d}.laz"))
        x, y = np.asarray(las.x), np.asarray(las.y)
        centros = centros_da_grade(polys[t])
        escritos = vazios = 0
        for i, (cx, cy) in enumerate(centros):
            m = (np.abs(x - cx) <= h) & (np.abs(y - cy) <= h)
            n = int(m.sum())
            if n < MIN_PTS:
                vazios += 1
                continue
            nome = f"t{t:03d}_w{i:04d}"
            # writes straight in the format SegmentAnyTree reads, instead of writing
            # the full tile and converting afterwards with convert_ff3d_tiles.py: there
            # are 1405 tiles and the conversion pass would cost a second copy on disk.
            # The rule is the same as there: point format 0, geometry only, ground
            # included. The attributes are dropped because their pandas_to_las.py fails
            # converting float32 to uint16 when writing the output.
            hdr = laspy.LasHeader(point_format=0, version="1.2")
            zz = np.asarray(las.z)[m]
            hdr.offsets = np.array([x[m].min(), y[m].min(), zz.min()])
            hdr.scales = np.array([0.001, 0.001, 0.001])
            sub = laspy.LasData(hdr)
            sub.x, sub.y, sub.z = x[m], y[m], zz
            sub.write(str(args.out / f"{nome}.laz"))
            escritos += 1
            linhas.append({"tile": nome, "talhao": t, "cx": cx, "cy": cy,
                           "n_pts": n,
                           "MB": round((args.out / f"{nome}.laz").stat().st_size / 1e6, 2)})
        print(f"  stand {t:03d}: {escritos} tiles written, {vazios} discarded for "
              f"having fewer than {MIN_PTS:,} points")
        del las, x, y

    man = pd.DataFrame(linhas)
    man.to_csv(args.out / "tiles_manifest.csv", index=False)
    print(f"\n{len(man)} tiles, {man.n_pts.sum():,} points, {man.MB.sum():.0f} MB in {args.out}")
    print(f"largest tile {man.MB.max():.1f} MB, median {man.MB.median():.1f} MB")
    print(f"prediction at 49 s/tile: {len(man)*49/3600:.1f} h")


if __name__ == "__main__":
    sys.exit(main())
