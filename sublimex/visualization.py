"""Visualization utilities for SublimeX."""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union


def _import_plt():
    import matplotlib.pyplot as plt
    return plt


def plot_feature_importance(importances, descriptions=None, top_k=None, 
                            figsize=(10, 6), color='#2ecc71'):
    """Plot feature importance as horizontal bar chart."""
    plt = _import_plt()
    n = len(importances)
    if descriptions is None:
        descriptions = [f'Feature {i+1}' for i in range(n)]
    
    indices = np.argsort(importances)[::-1]
    if top_k:
        indices = indices[:top_k]
    
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importances[indices], color=color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([descriptions[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    plt.tight_layout()
    return fig


def plot_segment_on_signal(signal, feature_params, n_time, transform_names=None,
                           sample_idx=0, figsize=(12, 4)):
    """Visualize where a feature's segment falls on the signal."""
    plt = _import_plt()
    sig = signal if signal.ndim == 1 else signal[sample_idx]
    
    c, r, t = feature_params.get('c', 0.5), feature_params.get('r', 0.5), feature_params.get('t', 0)
    center_idx = c * (n_time - 1)
    half_width = (r * (n_time - 1)) * 0.5
    start, end = max(0, int(center_idx - half_width)), min(n_time - 1, int(center_idx + half_width))
    
    transform_name = transform_names[t] if transform_names and t < len(transform_names) else f'transform_{t}'
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sig, color='#3498db', linewidth=1.5, label='Signal')
    ax.axvspan(start, end, alpha=0.3, color='#e74c3c', label=f'Segment [{start}:{end}]')
    ax.axvline(start, color='#e74c3c', linestyle='--', alpha=0.7)
    ax.axvline(end, color='#e74c3c', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'Feature segment (transform: {transform_name})')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_feature_distributions(features, y, feature_idx=0, feature_description=None,
                               figsize=(10, 5), bins=30):
    """Plot distribution of a feature's values across classes."""
    plt = _import_plt()
    feat_values = features[:, feature_idx]
    classes = np.unique(y)
    colors = plt.cm.get_cmap('tab10')(range(len(classes)))
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for i, cls in enumerate(classes):
        mask = y == cls
        axes[0].hist(feat_values[mask], bins=bins, alpha=0.6, color=colors[i], label=f'Class {cls}')
    axes[0].set_xlabel('Feature Value')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    
    bp = axes[1].boxplot([feat_values[y == c] for c in classes], 
                         labels=[f'Class {c}' for c in classes], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_ylabel('Feature Value')
    
    fig.suptitle(feature_description or f'Feature {feature_idx + 1}', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_transform_comparison(signal, transforms, figsize=(12, 8), sample_idx=0):
    """Show a signal under all available transforms."""
    plt = _import_plt()
    sig = signal.reshape(1, -1) if signal.ndim == 1 else signal[sample_idx:sample_idx+1]
    
    n_transforms = len(transforms)
    n_cols, n_rows = 2, (n_transforms + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_transforms))
    
    for i, (name, fn) in enumerate(transforms.items()):
        try:
            axes[i].plot(fn(sig)[0], color=colors[i], linewidth=1.5)
            axes[i].set_title(name, fontweight='bold')
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center', transform=axes[i].transAxes)
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Value')
    
    for i in range(n_transforms, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle('Signal Transforms Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_optimization_history(study_or_scores, figsize=(10, 5), color='#9b59b6'):
    """Plot optimization progress over trials."""
    plt = _import_plt()
    scores = [t.value for t in study_or_scores.trials if t.value is not None] if hasattr(study_or_scores, 'trials') else list(study_or_scores)
    
    fig, ax = plt.subplots(figsize=figsize)
    trials = np.arange(1, len(scores) + 1)
    ax.plot(trials, scores, alpha=0.3, color=color, label='Trial scores')
    ax.plot(trials, np.maximum.accumulate(scores), color=color, linewidth=2, label='Best so far')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Score')
    ax.set_title('Optimization Progress')
    ax.legend()
    plt.tight_layout()
    return fig
