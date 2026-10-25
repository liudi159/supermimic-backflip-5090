"""Portable paths and explicit single-GPU selection. No host settings are changed."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
B=Path(os.environ.get('BACKFLIP_WORKDIR',ROOT/'.runtime')).expanduser().resolve()
R=ROOT/'third_party/supermimic_vibed'
def configure_gpu():
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not gpu or ',' in gpu:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one GPU UUID (recommended) or index.')
    if not os.environ.get('MUJOCO_EGL_DEVICE_ID'):
        raise RuntimeError('Set MUJOCO_EGL_DEVICE_ID to that GPU physical index for EGL.')
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    return dict(os.environ)
