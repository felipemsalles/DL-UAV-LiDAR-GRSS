"""Session-aware test-time adaptation (TTA) for the FF3D detector.

Idea #7(b)-MODEL of the GreenVista IC: as a stand's tiles stream in during
deployment ("deployment-ordered stream"), adapt the *model internals* online,
unsupervised, so later tiles benefit from earlier ones. This is the
model-internal counterpart to the post-hoc merge-radius session adaptation.

The FF3D backbone (``SpConvUNet`` + the model's ``output_layer`` + the
``Embed`` / ``BiSemantic`` MLP heads) is built entirely from
``torch.nn.BatchNorm1d`` (running-stat BatchNorm) -> genuine BatchNorm-based
test-time adaptation is *applicable* here (unlike LayerNorm/GroupNorm nets, on
which TENT does not apply). The transformer *decoder* uses LayerNorm, which we
deliberately leave untouched.

Everything in this file is **inert unless the env flag ``FF3D_TTA_ADAPT`` is
set** to a recognised mode. When unset (the default), ``begin_session`` returns
``None`` before touching the model, so the production 6b/regen pipeline is
byte-for-byte unchanged.

Modes (env ``FF3D_TTA_ADAPT``)
------------------------------
- ``off`` / ``0`` / unset  -> disabled (default; no-op).
- ``adabn`` (or ``1``)     -> prediction-time BatchNorm (a.k.a. AdaBN /
  BN-stats adaptation). All BN layers are put in ``train()`` mode so each
  streamed region/tile is normalised by *its own* statistics instead of the
  frozen ForAINetV2 training statistics. No gradients, no disk state. Directly
  corrects the train(temperate forest)->test(eucalyptus UAV) covariate shift.
- ``session_ema``          -> the "deployment-ordered stream" realisation. BN
  stays in ``eval()`` (normalises with *running* stats), and forward-pre hooks
  fold each streamed region into the running stats via an EMA at rate
  ``FF3D_TTA_BN_MOMENTUM``. So region/tile ``k`` is normalised with statistics
  accumulated from all earlier regions/tiles, then contributes itself -> later
  tiles genuinely benefit from earlier ones. Because the inference driver
  (``inference_bluepoint_forestsens.sh``) spawns a *fresh process per tile*
  (see the design doc, "container persistence"), cross-tile memory is carried
  on **disk**: the accumulated running stats are saved to
  ``<state_dir>/ff3d_tta_<session>.pt`` at process end and reloaded at the next
  tile's process start (keyed by ``FF3D_TTA_SESSION``).
- ``tent``                 -> classic TENT (Wang 2021): entropy minimisation on
  the BN affine parameters (weight+bias only; everything else frozen), tiny LR.
  EXPERIMENTAL here -- see ``tent_adapt_from_points`` for the important caveat
  that the mmengine test loop runs under ``torch.no_grad()``, so the adaptation
  step performs its own ``enable_grad`` recompute. Adapted affine params are
  persisted per session like ``session_ema``.

Env knobs
---------
- ``FF3D_TTA_ADAPT``       mode (see above).
- ``FF3D_TTA_BN_MOMENTUM`` EMA rate for ``session_ema`` running-stat update
  (default 0.1).
- ``FF3D_TTA_SESSION``     session/stand id -> disk state filename. If unset in
  a session mode, it is derived from the scan filename prefix.
- ``FF3D_TTA_STATE_DIR``   directory for the per-session ``.pt`` state files
  (default: ``$FF3D_TTA_STATE_DIR`` or ``/tmp/ff3d_tta``).
- ``FF3D_TTA_RESET``       truthy -> ignore any persisted state (start a fresh
  stand session) and delete the session file on load.
- ``FF3D_TTA_LR``          TENT learning rate (default 1e-4).
- ``FF3D_TTA_VERBOSE``     truthy -> print a one-line summary per session.

None of this is wired into training; it only affects ``predict``.
"""

import os
import glob  # noqa: F401  (kept for potential multi-file session globbing)

import torch
from torch import nn

_FALSEY = ("", "0", "off", "false", "no", "none")
_ADABN_ALIASES = ("1", "true", "on", "yes", "adabn", "bn", "bn_stats")


def _env(name, default=""):
    return os.environ.get(name, default).strip()


def _truthy(name):
    return _env(name).lower() not in _FALSEY


def is_enabled():
    """True iff ``FF3D_TTA_ADAPT`` requests a real adaptation mode."""
    return _env("FF3D_TTA_ADAPT").lower() not in _FALSEY


def mode():
    """Normalised mode string: 'adabn' | 'session_ema' | 'tent' | 'off'."""
    raw = _env("FF3D_TTA_ADAPT").lower()
    if raw in _FALSEY:
        return "off"
    if raw in _ADABN_ALIASES:
        return "adabn"
    if raw in ("session_ema", "session", "ema"):
        return "session_ema"
    if raw in ("tent",):
        return "tent"
    # Unknown value -> be conservative and disable rather than surprise.
    return "off"


def _state_dir():
    d = _env("FF3D_TTA_STATE_DIR") or "/tmp/ff3d_tta"
    os.makedirs(d, exist_ok=True)
    return d


def _session_id(filename):
    sid = _env("FF3D_TTA_SESSION")
    if sid:
        return sid
    # Derive a stand-ish id from the scan name, e.g. "T1_2_..." -> "T1".
    base = os.path.basename(str(filename))
    stem = os.path.splitext(base)[0]
    return stem.split("_")[0] if stem else "default"


def _session_path(filename):
    return os.path.join(_state_dir(), f"ff3d_tta_{_session_id(filename)}.pt")


def _bn_modules(model):
    """All running-stat BatchNorm1d modules (backbone + heads).

    ``FastBatchNorm1d`` wraps an ``nn.BatchNorm1d`` as ``.batch_norm``; iterating
    ``model.modules()`` therefore also yields those inner BN layers, so a single
    ``isinstance`` filter captures every adaptable norm layer uniformly.
    """
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]


class TTAState:
    """Bag of per-session adaptation state; lives for one ``predict`` call."""

    def __init__(self, mode_, model, bns, filename):
        self.mode = mode_
        self.model = model
        self.bns = bns
        self.filename = filename
        self.handles = []          # forward-pre hook handles (session_ema)
        self.optimizer = None      # TENT optimiser
        self.momentum = float(_env("FF3D_TTA_BN_MOMENTUM") or 0.1)
        self.session_path = _session_path(filename)
        self.verbose = _truthy("FF3D_TTA_VERBOSE")
        self.n_tent_steps = 0

    def log(self, msg):
        if self.verbose:
            print(f"[FF3D-TTA:{self.mode}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# session_ema: online EMA of BN running statistics over the streamed regions   #
# --------------------------------------------------------------------------- #
def _make_ema_pre_hook(state):
    m = state.momentum

    def hook(module, inputs):
        x = inputs[0]
        if not torch.is_tensor(x) or x.numel() == 0 or x.dim() < 2:
            return
        if module.running_mean is None or module.running_var is None:
            return
        # Reduce every dim except the channel dim (1) -> per-channel stats.
        reduce_dims = [d for d in range(x.dim()) if d != 1]
        with torch.no_grad():
            batch_mean = x.mean(dim=reduce_dims)
            batch_var = x.var(dim=reduce_dims, unbiased=False)
            if batch_mean.shape != module.running_mean.shape:
                return  # unexpected layout; skip rather than corrupt
            module.running_mean.mul_(1 - m).add_(batch_mean, alpha=m)
            module.running_var.mul_(1 - m).add_(batch_var, alpha=m)

    return hook


def _load_session_stats(state):
    """Load persisted running stats (session_ema) / affine (tent) from disk."""
    path = state.session_path
    if _truthy("FF3D_TTA_RESET"):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        state.log(f"reset: ignoring/removing {path}")
        return False
    if not os.path.exists(path):
        state.log(f"no prior session state at {path} (first tile of stand)")
        return False
    try:
        blob = torch.load(path, map_location="cpu")
    except Exception as e:  # noqa: BLE001 - corrupt/partial file must not crash inference
        state.log(f"failed to load {path}: {e}; starting fresh")
        return False
    stats = blob.get("bn", [])
    n = min(len(stats), len(state.bns))
    for i in range(n):
        bn = state.bns[i]
        rm, rv = stats[i]
        if bn.running_mean is not None and rm.shape == bn.running_mean.shape:
            bn.running_mean.copy_(rm.to(bn.running_mean.device))
            bn.running_var.copy_(rv.to(bn.running_var.device))
    affine = blob.get("affine", None)
    if affine is not None:
        na = min(len(affine), len(state.bns))
        for i in range(na):
            bn = state.bns[i]
            w, b = affine[i]
            if bn.weight is not None and w.shape == bn.weight.shape:
                bn.weight.data.copy_(w.to(bn.weight.device))
                bn.bias.data.copy_(b.to(bn.bias.device))
    state.log(f"loaded session state for {n} BN layers from {path}")
    return True


def _save_session_stats(state):
    path = state.session_path
    blob = {
        "bn": [
            (bn.running_mean.detach().cpu().clone(),
             bn.running_var.detach().cpu().clone())
            for bn in state.bns
        ],
    }
    if state.mode == "tent":
        blob["affine"] = [
            (bn.weight.detach().cpu().clone(), bn.bias.detach().cpu().clone())
            for bn in state.bns
        ]
    try:
        torch.save(blob, path)
        state.log(f"saved session state ({len(state.bns)} BN layers) -> {path}")
    except Exception as e:  # noqa: BLE001
        state.log(f"failed to save session state to {path}: {e}")


# --------------------------------------------------------------------------- #
# TENT: entropy minimisation on BN affine params                              #
# --------------------------------------------------------------------------- #
def _configure_tent(state):
    lr = float(_env("FF3D_TTA_LR") or 1e-4)
    # Classic TENT: freeze the entire model, then re-enable ONLY the BN affine
    # params (weight+bias). Everything else stays frozen.
    for p in state.model.parameters():
        p.requires_grad_(False)
    params = []
    for bn in state.bns:
        bn.train()  # use & update batch stats during the adaptation forward
        if bn.weight is not None:
            bn.weight.requires_grad_(True)
            bn.bias.requires_grad_(True)
            params += [bn.weight, bn.bias]
    # OPTIMIZER IS CONFIGURABLE, AND THE REASON MATTERS. Wang 2021 p4 says to
    # follow "the training hyperparameters for the source model": Adam off
    # ImageNet, SGD with momentum on it. FF3D trains with AdamW, so the SGD
    # below is the ImageNet recipe and NOT this model's own. SegmentAnyTree
    # trains with Adam and its TENT uses Adam, so the two rows of the paper's
    # TENT column ran under different optimizers. FF3D_TENT_OPT=adam makes them
    # match. Default stays sgd so published runs reproduce bit for bit.
    qual = (_env("FF3D_TENT_OPT") or "sgd").strip().lower()
    if params:
        if qual == "adam":
            state.optimizer = torch.optim.Adam(params, lr=lr)
        else:
            state.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    state.log(f"TENT configured: {len(params)} affine tensors, lr={lr}, opt={qual}")


def _entropy(log_probs):
    """Mean Shannon entropy of a batch of *log*-probabilities (N, C)."""
    p = log_probs.exp()
    return -(p * log_probs).sum(dim=1).mean()


def tent_adapt_from_points(state, model, pc3):
    """One TENT step from a region's sampled points ``pc3`` (EXPERIMENTAL).

    The mmengine test loop wraps ``predict`` in ``@torch.no_grad()`` and the
    production forward converts to numpy immediately, so no autograd graph
    survives. To obtain a real gradient into the BN affine params without
    disturbing the (unchanged) prediction forward, this runs an independent
    ``enable_grad`` recompute of backbone -> ``BiSemantic`` on the same points
    and minimises the per-voxel wood/not-wood prediction entropy. Costs one
    extra backbone forward *in tent mode only*.
    """
    if state.optimizer is None:
        return
    import spconv.pytorch as spconv  # lazy: keeps this module CPU-importable
    with torch.enable_grad():
        coords, feats, inv_map, spatial_shape = model.collate([pc3])
        x = spconv.SparseConvTensor(feats, coords, spatial_shape, 1)
        x = model.extract_feat(x)
        log_probs = model.BiSemantic(x[0])          # LogSoftmax over 2 classes
        loss = _entropy(log_probs)
        state.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # ⚠️ MEDIR O PASSO, e nao so a perda. "Adam vs SGD com o MESMO lr" nao
        # compara otimizadores, compara deslocamentos: o passo do Adam e
        # aproximadamente lr por iteracao qualquer que seja o gradiente, e o do
        # SGD e lr*grad amplificado pelo momento. Se o gradiente da entropia for
        # pequeno, o braco SGD quase nao adapta e a diferenca medida no F1 nao e
        # do otimizador. |grad| e |delta| abaixo respondem isso direto, sem
        # depender do desfecho.
        grupo = state.optimizer.param_groups[0]["params"]
        gn = torch.sqrt(sum((p.grad.detach() ** 2).sum()
                            for p in grupo if p.grad is not None))
        antes = [p.detach().clone() for p in grupo]
        state.optimizer.step()
        dn = torch.sqrt(sum(((p.detach() - a) ** 2).sum()
                            for p, a in zip(grupo, antes)))
    state.n_tent_steps += 1
    state.log(f"tent step {state.n_tent_steps}: entropy={float(loss):.4f} "
              f"|grad|={float(gn):.3e} |delta|={float(dn):.3e}")


# --------------------------------------------------------------------------- #
# public entry points (called, guarded, from ForAINetV2..._XAwarequery.predict) #
# --------------------------------------------------------------------------- #
def begin_session(model, filename):
    """Configure TTA for one ``predict`` call. Returns ``None`` when disabled.

    Safe to call unconditionally from the model; it re-checks the env flag and
    returns immediately (touching nothing) when adaptation is off.
    """
    m = mode()
    if m == "off":
        return None
    bns = _bn_modules(model)
    if not bns:  # no BatchNorm -> nothing model-internal to adapt
        return None
    state = TTAState(m, model, bns, filename)

    if m == "adabn":
        for bn in bns:
            bn.train()  # normalise each streamed region by its own stats
        state.log(f"AdaBN active on {len(bns)} BN layers")

    elif m == "session_ema":
        for bn in bns:
            bn.eval()  # normalise with running stats; hooks fold in new data
        _load_session_stats(state)
        for bn in bns:
            state.handles.append(bn.register_forward_pre_hook(
                _make_ema_pre_hook(state)))
        state.log(f"session EMA active (momentum={state.momentum}) on "
                  f"{len(bns)} BN layers")

    elif m == "tent":
        _load_session_stats(state)
        _configure_tent(state)

    return state


def end_session(state):
    """Tear down hooks and persist session state. No-op if ``state`` is None."""
    if state is None:
        return
    for h in state.handles:
        h.remove()
    state.handles = []
    if state.mode in ("session_ema", "tent"):
        _save_session_stats(state)
