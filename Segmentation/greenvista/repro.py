"""Reproducibility + lightweight experiment tracking.

`seed_everything` pins the Python/NumPy/PyTorch RNGs and CUDA determinism (cuDNN deterministic, no
benchmark autotune, deterministic algorithms) so GPU runs are bit-reproducible; at N=13 a single fold
swing moves R², so runs have to be pinned.

`write_manifest` persists a JSON run manifest (git SHA, timestamp, config, seed, metrics) next to
each result CSV, so every table is traceable to a commit + config.
"""
import json
import os
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path


def seed_everything(seed: int = 0, deterministic: bool = True) -> int:
    """Seed random / numpy / torch (+ CUDA) and, if `deterministic`, force reproducible kernels."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required for deterministic cublas
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass
    return seed


def git_sha(default: str = "unknown") -> str:
    """Current commit SHA (for provenance), or `default` outside a repo."""
    try:
        return (subprocess.check_output(["git", "rev-parse", "HEAD"],
                                        cwd=str(Path(__file__).resolve().parent),
                                        stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        return default


def _jsonable(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def write_manifest(path, *, script: str, config, metrics, seed: int = 0, extra: dict | None = None) -> dict:
    """Write a JSON run manifest to `path` and return it. `config` may be a dataclass or dict."""
    man = {
        "script": script,
        "git_sha": git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "seed": seed,
        "config": _jsonable(config),
        "metrics": _jsonable(metrics),
    }
    if extra:
        man.update(_jsonable(extra))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, indent=2, default=str))
    return man
