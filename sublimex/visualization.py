"""Minimal viz: feature importance bar, segment on signal."""
import numpy as np
import matplotlib.pyplot as plt

def plot_segment_on_signal(signal, feature_params, n_time, sample_idx=0, figsize=(10, 3)):
    sig = signal[sample_idx] if signal.ndim > 1 else signal
    c, r = feature_params.get('c', 0.5), feature_params.get('r', 0.5)
    center = c * (n_time - 1)
    half = (r * (n_time - 1)) * 0.5
    start = max(0, int(center - half))
    end = min(n_time - 1, int(center + half))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sig, color='#3498db', lw=1.5)
    ax.axvspan(start, end, alpha=0.3, color='#e74c3c')
    ax.set_xlabel('Time')
    plt.tight_layout()
    return fig
