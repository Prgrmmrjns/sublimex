"""Signal transforms: (..., n_time) -> same shape."""
import numpy as np


def _fft_power(d):
    p = np.abs(np.fft.rfft(d, axis=-1)) ** 2
    x = np.linspace(0, p.shape[-1] - 1, d.shape[-1])
    i = np.minimum(x.astype(int), p.shape[-1] - 2)
    return (p[..., i] * (1 - (x - i)) + p[..., i + 1] * (x - i)).astype(np.float32)


TRANSFORMS = {
    'raw': lambda d: d.astype(np.float32),
    'zscore': lambda d: ((d - d.mean(-1, keepdims=True)) / (d.std(-1, keepdims=True) + 1e-8)).astype(np.float32),
    'derivative': lambda d: np.gradient(d, axis=-1).astype(np.float32),
    'fft_power': _fft_power,
}
