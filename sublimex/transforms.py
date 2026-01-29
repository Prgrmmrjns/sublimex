"""Signal transforms for SublimeX feature extraction."""
import numpy as np
from typing import Callable, Dict, List


def _fft_power(d):
    """FFT power spectrum interpolated to original length."""
    p = np.abs(np.fft.rfft(d, axis=-1)) ** 2
    x = np.linspace(0, p.shape[-1]-1, d.shape[-1])
    idx = np.minimum(x.astype(int), p.shape[-1]-2)
    return p[..., idx] * (1 - (x - idx)) + p[..., idx + 1] * (x - idx)


TRANSFORMS: Dict[str, Callable] = {
    'raw': lambda d: d,
    'zscore': lambda d: (d - d.mean(-1, keepdims=True)) / (d.std(-1, keepdims=True) + 1e-8),
    'derivative': lambda d: np.gradient(d, axis=-1),
    'fft_power': _fft_power,
}


def register_transform(name: str, func: Callable, overwrite: bool = False):
    """Register a custom transform function."""
    if name in TRANSFORMS and not overwrite:
        raise ValueError(f"Transform '{name}' already exists. Use overwrite=True.")
    TRANSFORMS[name] = func


def get_transform(name: str) -> Callable:
    """Get transform by name."""
    if name not in TRANSFORMS:
        raise KeyError(f"Transform '{name}' not found. Available: {list(TRANSFORMS.keys())}")
    return TRANSFORMS[name]


def list_transforms() -> List[str]:
    """List available transform names."""
    return list(TRANSFORMS.keys())
