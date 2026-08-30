"""Leave-one-tree-out CV for per-tree DAP: allometry baseline vs CNN / PointNet / PointTransformer.

Everything is judged by leave-one-tree-out CV (N=23). Because deep nets can fit noise at this N,
each net is also run once on shuffled labels — a control that must collapse to R²≈0; if the real run
is not clearly above the shuffled one, the apparent signal is overfitting. Targets are standardized
in log space per fold (train-only); test predictions use test-time augmentation (rotation
averaging).
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from greenvista.repro import seed_everything
from greenvista.eval import metrics, permutation_test
from greenvista.image_cnn import render
from . import data as _d
from .models import DEEP_MODELS, DapAllometry


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_compatible(net, state):
    """Copy only params whose name+shape match (transfers the SSL backbone, skips the pretext head).
    Returns (kept, total)."""
    own = net.state_dict()
    keep = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(keep)
    net.load_state_dict(own)
    return len(keep), len(own)


def _rotate_about(pts, center, th):
    p = pts.copy()
    x, y = p[:, 0] - center[0], p[:, 1] - center[1]
    ct, st = np.cos(th), np.sin(th)
    p[:, 0] = center[0] + x * ct - y * st
    p[:, 1] = center[1] + x * st + y * ct
    return p


def _raster(tree, size, radius, rng, augment):
    pts = tree["points"]
    if augment:
        pts = _rotate_about(pts, tree["center"], rng.uniform(0, 2 * np.pi))
    r, _ = render.to_raster(pts, tuple(tree["center"]), radius, size=size)
    return r.astype(np.float32)


def _point(tree, n_sample, rng, augment):
    return _d.normalize_points(tree["points"], tree["center"], n_sample, rng, augment=augment)


def _batch(trees, arvs, kind, rng, augment, n_sample, size, radius):
    if kind == "raster":
        X = np.stack([_raster(trees[a], size, radius, rng, augment) for a in arvs])
    else:
        X = np.stack([_point(trees[a], n_sample, rng, augment) for a in arvs])
    return torch.from_numpy(X)


def _loocv_deep(trees, model_name, dap, *, n_sample=512, size=32, radius=2.0, epochs=40,
                n_aug=6, lr=1e-3, seed=0, tta=8, dev=None, shuffle=False, amp=True,
                init_state=None, freeze_backbone=False):
    """Leave-one-tree-out predictions for a deep model. `shuffle=True` permutes labels (control).

    `amp=True` runs the forward/backward in mixed precision (torch.autocast + GradScaler) on CUDA —
    ~1.5-2x faster with no result change; ignored on CPU. `init_state` (from SSL pretraining) warm-
    starts the backbone each fold; `freeze_backbone` trains only the head (linear-probe)."""
    dev = dev or device()
    use_amp = amp and dev.type == "cuda"
    kind = "raster" if model_name == "CNN" else "point"
    arvs = sorted(trees)
    y_all = np.array([dap[a] for a in arvs], float)
    if shuffle:
        y_all = np.random.default_rng(seed + 999).permutation(y_all)
    pred = np.empty(len(arvs))
    for i, test_arv in enumerate(arvs):
        seed_everything(seed)
        tr = [a for a in arvs if a != test_arv]
        ytr = np.log(np.array([y_all[arvs.index(a)] for a in tr], float))
        ym, ys = ytr.mean(), ytr.std() + 1e-6
        rng = np.random.default_rng(seed)
        net = DEEP_MODELS[model_name]().to(dev)
        if init_state is not None:                       # SSL warm-start (backbone transfer)
            _load_compatible(net, init_state)
            if freeze_backbone and hasattr(net, "mlp"):
                for p in net.mlp.parameters():
                    p.requires_grad_(False)
        opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        lossf = torch.nn.MSELoss()
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32, device=dev)
        net.train()
        for _ in range(epochs):
            for _ in range(n_aug):
                X = _batch(trees, tr, kind, rng, True, n_sample, size, radius).to(dev)
                opt.zero_grad()
                with torch.autocast("cuda", enabled=use_amp):
                    loss = lossf(net(X), yt)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", enabled=use_amp):
            views = [net(_batch(trees, [test_arv], kind, rng, k > 0, n_sample, size, radius).to(dev)).item()
                     for k in range(max(1, tta))]
        pred[i] = np.exp(np.mean(views) * ys + ym)
    return y_all, pred


def _loocv_allometry(trees, dap, cols=("zmax", "crown_area")):
    arvs = sorted(trees)
    feats = [_d.crown_scalars(trees[a]["points"]) for a in arvs]
    y = np.array([dap[a] for a in arvs], float)
    pred = np.empty(len(arvs))
    for i in range(len(arvs)):
        tr = [j for j in range(len(arvs)) if j != i]
        m = DapAllometry(cols).fit([feats[j] for j in tr], y[tr])
        pred[i] = m.predict([feats[i]])[0]
    return y, pred


def evaluate_all(trees, seeds=(0, 1, 2), parallel=True, max_workers=3, **kw):
    """Return {model: metrics} under identical LOOCV, plus each net's shuffled-label control R².

    `parallel=True` runs the independent seed LOOCVs concurrently in a thread pool — the CPU-side
    augmentation (scipy rotation / point sampling releases the GIL) overlaps with GPU compute, so the
    multi-seed wall-time drops (~1.3x on one GPU; more with a bigger GPU or larger models). AMP
    (`amp=True` in kw) helps compute-bound nets but is ~neutral on these tiny ones.

    Caveat: parallel threads share the global torch RNG (dropout/init interleave), so
    `parallel=True` is not bit-reproducible. The seed spread is still valid for variance estimation,
    but an exactly reproducible run needs `parallel=False`."""
    dap = {a: trees[a]["D_cm"] for a in trees}
    out = {}

    for name, cols in {"allometry zmax": ("zmax",),
                       "allometry zmax+area": ("zmax", "crown_area"),
                       "allometry zmax+zmean+area": ("zmax", "zmean", "crown_area")}.items():
        y, p = _loocv_allometry(trees, dap, cols)
        out[name] = {**metrics(y, p), "perm_p": permutation_test(y, p, n_perm=3000)["p_value"]}

    def run_seed(model_name, s, shuffle=False):
        return _loocv_deep(trees, model_name, dap, seed=s, shuffle=shuffle, **kw)

    for model_name in DEEP_MODELS:
        jobs = list(seeds) + ["shuffle"]
        if parallel and device().type == "cuda":
            with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as ex:
                res = list(ex.map(lambda j: run_seed(model_name, 0 if j == "shuffle" else j,
                                                     shuffle=(j == "shuffle")), jobs))
        else:
            res = [run_seed(model_name, 0 if j == "shuffle" else j, shuffle=(j == "shuffle")) for j in jobs]
        seed_res = res[:len(seeds)]
        ys, ps = res[-1]                                 # shuffled control
        r2s = [metrics(y, p)["R2"] for y, p in seed_res]
        y = seed_res[0][0]
        preds = np.mean([p for _, p in seed_res], axis=0)
        out[model_name] = {**metrics(y, preds), "R2_seed_mean": float(np.mean(r2s)),
                           "R2_seed_sd": float(np.std(r2s)), "R2_shuffled": metrics(ys, ps)["R2"],
                           "perm_p": permutation_test(y, preds, n_perm=3000)["p_value"]}
    return out
