"""Leave-one-plot-out training/eval for the hybrid CNN (N=13).

Per fold: standardize rasters/scalars/targets on the **training plots only**, train the CNN on
augmented training rasters, predict the single held-out plot. Prediction intervals come from the
training-residual std (normal PI), matching `area_based.validate`. Nothing about the held-out plot
touches training — augmentation lives inside the train Dataset, normalization stats are train-only.
"""
import numpy as np
import torch
from scipy.ndimage import rotate as nd_rotate
from torch.utils.data import DataLoader

from greenvista.eval import metrics, bootstrap_ci, jackknife_plus_interval, multiseed_summary
from greenvista.repro import seed_everything
from .dataset import PlotRasterDataset, lopo_splits, _circle_mask
from .model import PlotCNN


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Norm:
    """Standardize rasters (per channel), scalars, and targets using train-fold stats only.

    Global standardization (fixed train stats applied to every plot) preserves the between-plot
    height differences that drive volume; it is not the per-plot min-max normalization that
    `render.to_raster` deliberately avoids.
    """
    def fit(self, samples, train_ids):
        R = np.stack([samples[p]["raster"] for p in train_ids])      # (n,C,H,W)
        self.cm = R.mean(axis=(0, 2, 3)); self.cs = R.std(axis=(0, 2, 3)) + 1e-6
        S = np.stack([samples[p]["scalars"] for p in train_ids])
        self.sm = S.mean(0); self.ss = S.std(0) + 1e-6
        Y = np.stack([samples[p]["labels"] for p in train_ids])
        self.ym = Y.mean(0); self.ys = Y.std(0) + 1e-6
        return self

    def nr(self, r):  # raster (B,C,H,W) tensor
        cm = torch.tensor(self.cm, dtype=r.dtype, device=r.device)[None, :, None, None]
        cs = torch.tensor(self.cs, dtype=r.dtype, device=r.device)[None, :, None, None]
        return (r - cm) / cs

    def ns(self, s):  # scalars (B,S)
        return (s - torch.tensor(self.sm, dtype=s.dtype, device=s.device)) / \
               torch.tensor(self.ss, dtype=s.dtype, device=s.device)

    def ny(self, y):  # targets (B,T) -> standardized
        return (y - torch.tensor(self.ym, dtype=y.dtype, device=y.device)) / \
               torch.tensor(self.ys, dtype=y.dtype, device=y.device)

    def iny(self, y_std):  # standardized -> raw target units
        return y_std * torch.tensor(self.ys, dtype=y_std.dtype, device=y_std.device) + \
               torch.tensor(self.ym, dtype=y_std.dtype, device=y_std.device)


def _load_compatible(net, state):
    """Load only the parameters whose names+shapes match (so a pretext head of different width — e.g.
    an SSL backbone that predicted K metrics — transfers its conv backbone without a shape error)."""
    own = net.state_dict()
    keep = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(keep); net.load_state_dict(own)
    return len(keep), len(own)


def _train_one(samples, train_ids, nrm, dev, n_aug, epochs, lr, seed, init_state=None,
               freeze_backbone=False):
    seed_everything(seed)                            # pins python/numpy/torch + cuDNN determinism
    ds = PlotRasterDataset(samples, train_ids, train=True, n_aug=n_aug, seed=seed)
    dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=False)
    net = PlotCNN().to(dev)
    if init_state is not None:                       # transfer: warm-start from pretrained backbone
        kept, total = _load_compatible(net, init_state)
        if freeze_backbone:                          # linear probe — train only the head at N=13
            for p in net.features.parameters():
                p.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    lossf = torch.nn.MSELoss()
    net.train()
    for _ in range(epochs):
        for r, s, y, _pid in dl:
            r, s, y = r.to(dev), s.to(dev), y.to(dev)
            opt.zero_grad()
            pred = net(nrm.nr(r), nrm.ns(s))
            loss = lossf(pred, nrm.ny(y))
            loss.backward(); opt.step()
    return net


def pretrain(samples, n_aug=2, epochs=20, lr=1e-3, seed=0, dev=None):
    """Pretrain a PlotCNN on (simulated) samples. Returns a CPU state_dict to warm-start LOPO folds.

    Standardization is fit on the sim samples; because both sim and real inputs are standardized to
    ~unit scale, the warm-started weights transfer cleanly to the real-normalized fine-tune folds.
    """
    dev = dev or device()
    ids = list(samples)
    nrm = _Norm().fit(samples, ids)
    net = _train_one(samples, ids, nrm, dev, n_aug, epochs, lr, seed)
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}


def _tta_views(raster, k, mask):
    """k rotation-augmented copies of a raster (identity + evenly spaced angles), circle-masked.
    Label-preserving for a nadir circular footprint, so averaging predictions over them is a valid
    variance reduction at eval time."""
    views = [np.ascontiguousarray(raster, np.float32)]
    for a in np.linspace(0, 360, k, endpoint=False)[1:]:
        out = nd_rotate(raster, float(a), axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0)
        views.append(np.ascontiguousarray(np.where(mask, out, 0.0), np.float32))
    return views


@torch.no_grad()
def _predict(net, samples, ids, nrm, dev, tta=0):
    net.eval()
    ev = PlotRasterDataset(samples, ids, train=False)
    n_t = len(ev[0][2])
    out = np.zeros((len(ids), n_t), np.float32)
    mask = _circle_mask(ev[0][0].shape[1]) if tta else None
    for i in range(len(ev)):
        r, s, _y, _pid = ev[i]
        s_dev = nrm.ns(s[None].to(dev))
        if tta and tta > 1:                          # average predictions over TTA rotations
            preds = []
            for v in _tta_views(r.numpy(), tta, mask):
                rt = torch.from_numpy(v)[None].to(dev)
                preds.append(nrm.iny(net(nrm.nr(rt), s_dev))[0].cpu().numpy())
            out[i] = np.mean(preds, axis=0)
        else:
            p = net(nrm.nr(r[None].to(dev)), s_dev)
            out[i] = nrm.iny(p)[0].cpu().numpy()
    return out


def run_lopo(samples, plot_ids, y_true, n_aug=64, epochs=80, lr=1e-3, seed=0, dev=None,
             verbose=False, init_state=None, tta=0, freeze_backbone=False):
    """Leave-one-plot-out CNN predictions + 95% PI for every plot.

    `y_true` is (n, n_targets) aligned to `plot_ids` (columns [Vol_ha, D_cm]). Returns
    {pred, lo, hi, transfer} each (n, n_targets), in raw target units. `init_state` (from `pretrain`)
    warm-starts every fold for transfer learning — the sim/SSL it came from never saw any real plot,
    so LOPO stays valid. `tta` averages predictions over rotations; `freeze_backbone` turns the
    transfer into a linear probe of the frozen backbone. Reported coverage uses the conformal
    interval (see `report`); the returned lo/hi remain the parametric band for the map.
    """
    dev = dev or device()
    plot_ids = list(plot_ids)
    n, T = len(plot_ids), y_true.shape[1]
    pred = np.zeros((n, T), np.float32); lo = np.zeros_like(pred); hi = np.zeros_like(pred)
    z = 1.959963985
    transfer = None
    for i, (test_id, train_ids) in enumerate(lopo_splits(plot_ids)):
        nrm = _Norm().fit(samples, train_ids)
        net = _train_one(samples, train_ids, nrm, dev, n_aug, epochs, lr, seed,
                         init_state=init_state, freeze_backbone=freeze_backbone)
        if init_state is not None and transfer is None:
            transfer = _load_compatible(PlotCNN().to(dev), init_state)  # (kept, total) for the log
        pred[i] = _predict(net, samples, [test_id], nrm, dev, tta=tta)[0]
        # training-residual std (un-augmented train predictions) → normal PI (map band only)
        tr_pred = _predict(net, samples, train_ids, nrm, dev)
        tr_true = np.stack([samples[p]["labels"] for p in train_ids])
        resid_sd = (tr_true - tr_pred).std(axis=0, ddof=1)
        lo[i] = pred[i] - z * resid_sd; hi[i] = pred[i] + z * resid_sd
        if verbose:
            print(f"  fold {i+1:2d}/{n} held-out {test_id}: "
                  f"Vol pred={pred[i,0]:.0f} true={y_true[i,0]:.0f}")
    if verbose and transfer is not None:
        print(f"  [transfer] warm-started {transfer[0]}/{transfer[1]} tensors "
              f"({'linear-probe' if freeze_backbone else 'full fine-tune'})")
    return {"pred": pred, "lo": lo, "hi": hi, "transfer": transfer}


def run_lopo_multiseed(samples, plot_ids, y_true, seeds=(0, 1, 2, 3, 4), col=0, **kw):
    """Run LOPO over several seeds and summarize R²/rRMSE as mean ± sd — a single-seed R² at N=13 is
    high-variance, so this reports the spread. Returns {per_seed, summary, best}."""
    per_seed = []
    for s in seeds:
        res = run_lopo(samples, plot_ids, y_true, seed=s, **kw)
        rep = report(y_true, res["pred"], res["lo"], res["hi"], col=col)
        per_seed.append({"seed": s, **rep, "_res": res})
    summary = {**multiseed_summary(per_seed, "R2"), **multiseed_summary(per_seed, "rRMSE_pct")}
    best = max(per_seed, key=lambda d: d["R2"])
    return {"per_seed": [{k: v for k, v in d.items() if k != "_res"} for d in per_seed],
            "summary": summary, "best": {k: v for k, v in best.items() if k != "_res"},
            "best_res": best["_res"]}


def report(y_true, pred, lo, hi, col=0):
    """metrics + bootstrap CI + conformal 95% coverage for one target column (default Vol_ha).

    Coverage is computed from the out-of-fold conformal interval — the single coverage source shared
    with the classical ladder — and not from the parametric train-residual band that `lo`/`hi` carry
    (those remain for the map). `lo`/`hi` are still accepted for signature stability.
    """
    y, p = y_true[:, col], pred[:, col]
    mm = metrics(y, p)
    ci = bootstrap_ci(y, p)
    mm["R2_ci"] = ci["R2_ci"]; mm["rRMSE_ci"] = ci["rRMSE_ci"]
    c_lo, c_hi = jackknife_plus_interval(y, p, alpha=0.05)
    mm["coverage95"] = float(((y >= c_lo) & (y <= c_hi)).mean())
    mm["coverage95_parametric"] = float(((y >= lo[:, col]) & (y <= hi[:, col])).mean())
    return mm


def fit_full(samples, plot_ids, n_aug=64, epochs=80, lr=1e-3, seed=0, dev=None, init_state=None):
    """Train one model on all given plots (for the wall-to-wall map). Returns (net, _Norm, resid_sd).

    resid_sd is the in-sample residual std per target → a (loose) interval width for map cells.
    """
    dev = dev or device()
    plot_ids = list(plot_ids)
    nrm = _Norm().fit(samples, plot_ids)
    net = _train_one(samples, plot_ids, nrm, dev, n_aug, epochs, lr, seed, init_state=init_state)
    pred = _predict(net, samples, plot_ids, nrm, dev)
    true = np.stack([samples[p]["labels"] for p in plot_ids])
    return net, nrm, (true - pred).std(axis=0, ddof=1)


def conformal_interval(y_true, pred, alpha=0.05):
    """Leave-one-out (jackknife+) conformal interval. Thin alias for
    `greenvista.eval.conformal.jackknife_plus_interval`, which applies the finite-sample quantile
    level ceil((1-α)(m+1))/m; the plain (1-α) quantile under-covers at N=13."""
    return jackknife_plus_interval(y_true, pred, alpha=alpha)


@torch.no_grad()
def predict_raster(net, nrm, raster, scalars, dev=None):
    """Predict raw-unit targets for a single rendered window (raster (C,H,W), scalars (S,))."""
    dev = dev or device()
    net.eval()
    r = torch.from_numpy(np.ascontiguousarray(raster, np.float32))[None].to(dev)
    s = torch.from_numpy(np.asarray(scalars, np.float32))[None].to(dev)
    return nrm.iny(net(nrm.nr(r), nrm.ns(s)))[0].cpu().numpy()
