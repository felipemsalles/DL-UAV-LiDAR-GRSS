#!/bin/bash
# =============================================================================
# GreenVista IC - Environment Setup Script
# RTX 5060 (Blackwell SM 120) + CUDA 12.8 + WSL2
#
# This env is for data processing, ML (XGBoost), and visualization.
# Segmentation models (ForestFormer3D, SegmentAnyTree) run in Docker.
# =============================================================================
set -euo pipefail

ENV_NAME="greenvista"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  GreenVista Environment Setup"
echo "  Target: RTX 5060 (Blackwell) + CUDA 12.8"
echo "============================================"

# --- Step 1: Create conda environment ---
echo ""
echo "[1/3] Creating conda environment '${ENV_NAME}' from environment.yml..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Environment '${ENV_NAME}' already exists. Removing..."
    conda deactivate 2>/dev/null || true
    mamba env remove -n "${ENV_NAME}" -y
fi
mamba env create -f "${SCRIPT_DIR}/environment.yml"

# --- Step 2: Verify PyTorch + CUDA ---
echo ""
echo "[2/3] Verifying PyTorch + CUDA..."
conda run -n "${ENV_NAME}" python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version:    {torch.version.cuda}')
    print(f'  GPU:             {torch.cuda.get_device_name(0)}')
    print(f'  GPU memory:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'  Arch:            SM {torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}')
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print(f'  Compute test:    OK')
else:
    print('  WARNING: CUDA not available! Check driver/toolkit.')
"

# --- Step 3: Full verification ---
echo ""
echo "[3/3] Verifying all packages..."
conda run -n "${ENV_NAME}" python -c "
import sys
print(f'  Python:       {sys.version.split()[0]}')

modules = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'torchaudio': 'TorchAudio',
    'open3d': 'Open3D',
    'laspy': 'laspy',
    'sklearn': 'scikit-learn',
    'xgboost': 'XGBoost',
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'shap': 'SHAP',
    'h5py': 'h5py',
    'plyfile': 'plyfile',
    'tensorboard': 'TensorBoard',
    'seaborn': 'seaborn',
    'rich': 'rich',
    'tqdm': 'tqdm',
    'wandb': 'W&B',
    'pyntcloud': 'pyntcloud',
}

ok = 0
fail = 0
for mod, name in modules.items():
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', 'OK')
        print(f'  {name:15s} {ver}')
        ok += 1
    except ImportError as e:
        print(f'  {name:15s} MISSING ({e})')
        fail += 1

print(f'\n  {ok}/{ok+fail} packages OK')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Activate with: conda activate ${ENV_NAME}"
echo ""
echo "  NOTE: ForestFormer3D and SegmentAnyTree"
echo "  run in Docker (they need old PyTorch/CUDA)."
echo "  This env is for processing + ML."
echo "============================================"
