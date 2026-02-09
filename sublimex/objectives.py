"""Objective functions for SublimeX optimization.

Each objective has signature (trial, ctx) -> float. Optuna's trial is used to
suggest parameters (e.g. channel, transform index, segment center/range).
The context dict `ctx` is provided by SublimeX and contains:

  - transformed : ndarray shape (n_transforms, n_samples, n_channels, n_time)
  - y : target array
  - model : object with .evaluate(X_train, y_train, X_val, y_val, metric)
  - metric : str ('auc', 'accuracy', 'rmse')
  - n_channels, n_time : int
  - transform_names : list of str
  - cv_splits : list of (train_idx, val_idx)
  - current_X : ndarray (n_samples, n_features_so_far) already selected features
  - extract_only : bool; if True, only compute and set ctx['last_feature'], return 0.0

Your objective should: suggest segment params from trial, extract a single feature
vector (n_samples, 1), set ctx['last_feature'] = feature when in extract_only mode,
and otherwise return the mean CV score from _evaluate(feature, ctx).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Callable, Dict, Any


AGGREGATIONS = {
    'mean': lambda x: x.mean(axis=1, keepdims=True),
    'min': lambda x: x.min(axis=1, keepdims=True),
    'max': lambda x: x.max(axis=1, keepdims=True),
    'range': lambda x: np.ptp(x, axis=1, keepdims=True),
    'std': lambda x: x.std(axis=1, keepdims=True),
    'median': lambda x: np.median(x, axis=1, keepdims=True),
    'argmin': lambda x: x.argmin(axis=1, keepdims=True).astype(np.float32) / max(x.shape[1] - 1, 1),
    'argmax': lambda x: x.argmax(axis=1, keepdims=True).astype(np.float32) / max(x.shape[1] - 1, 1),
}
AGG_KEYS = list(AGGREGATIONS.keys())


def get_segment_indices(center: float, range_val: float, n_time: int) -> tuple:
    """Convert normalized center and range to segment indices."""
    center_idx = center * (n_time - 1)
    half_width = (range_val * (n_time - 1)) * 0.5
    start = max(0, int(center_idx - half_width))
    end = min(n_time - 1, int(center_idx + half_width))
    return start, end


def _evaluate(feature: np.ndarray, ctx: Dict[str, Any]) -> float:
    """Evaluate a candidate feature using cross-validation."""
    ctx['last_feature'] = feature
    if ctx.get('extract_only'):
        return 0.0
    
    current_X = ctx['current_X']
    X = np.hstack([current_X, feature]) if current_X.size else feature
    
    scores = []
    for train_idx, val_idx in ctx['cv_splits']:
        score = ctx['model'].evaluate(X[train_idx], ctx['y'][train_idx], 
                                      X[val_idx], ctx['y'][val_idx], ctx['metric'])
        scores.append(score)
    return np.mean(scores)


def _get_segment(trial, ctx: Dict[str, Any], feature_type: str) -> tuple:
    """Extract segment parameters from trial and retrieve data."""
    ch = trial.suggest_int('ch', 0, ctx['n_channels'] - 1)
    t = trial.suggest_int('t', 0, len(ctx['transform_names']) - 1)
    c = trial.suggest_float('c', 0, 1)
    r = trial.suggest_float('r', 0, 1)
    
    start, end = get_segment_indices(c, r, ctx['n_time'])
    segment = ctx['transformed'][t, :, ch, start:end+1]
    
    params = {
        'feature_type': feature_type, 'channel': ch, 'transform_idx': t,
        'transform_name': ctx['transform_names'][t], 'center': c, 'range': r,
        'start_idx': start, 'end_idx': end,
    }
    return segment, params


def mean_objective(trial, ctx: Dict[str, Any]) -> float:
    """Mean objective - extract mean value over optimized segment."""
    segment, _ = _get_segment(trial, ctx, 'mean')
    feature = segment.mean(axis=1, keepdims=True).astype(np.float32)
    return _evaluate(feature, ctx)


def aggregate_objective(trial, ctx: Dict[str, Any]) -> float:
    """Aggregate objective - choose from multiple aggregation functions."""
    segment, _ = _get_segment(trial, ctx, 'aggregate')
    agg_name = trial.suggest_categorical('agg', AGG_KEYS)
    feature = AGGREGATIONS[agg_name](segment).astype(np.float32)
    return _evaluate(feature, ctx)


def pattern_objective(trial, ctx: Dict[str, Any]) -> float:
    """Pattern objective - quadratic B-spline matching."""
    segment, params = _get_segment(trial, ctx, 'pattern')
    
    w = trial.suggest_float('w', 0.05, 0.5)
    cp0 = trial.suggest_float('cp0', 0, 1)
    cp1 = trial.suggest_float('cp1', 0, 1)
    cp2 = trial.suggest_float('cp2', 0, 1)
    
    n_time = ctx['n_time']
    n_samples, segment_len = segment.shape
    width = min(max(2, int(w * n_time)), segment_len)
    
    if width > segment_len or segment_len - width + 1 <= 0:
        return _evaluate(np.full((n_samples, 1), np.inf, dtype=np.float32), ctx)
    
    t = np.linspace(0, 1, width, dtype=np.float32)
    pattern = (1 - t) ** 2 * cp0 + 2 * (1 - t) * t * cp1 + t ** 2 * cp2
    
    windows = sliding_window_view(segment, window_shape=width, axis=1)
    distances = np.linalg.norm(windows - pattern, axis=2) / np.sqrt(width)
    feature = distances.min(axis=1, keepdims=True).astype(np.float32)
    
    return _evaluate(feature, ctx)


default_objective = mean_objective


def create_custom_objective(aggregation_fn: Callable, name: str = 'custom') -> Callable:
    """Create a custom objective with user-defined aggregation."""
    def custom_objective(trial, ctx: Dict[str, Any]) -> float:
        segment, _ = _get_segment(trial, ctx, name)
        feature = aggregation_fn(segment).astype(np.float32)
        return _evaluate(feature, ctx)
    custom_objective.__name__ = f'{name}_objective'
    return custom_objective


def parallel_objective(trial, ctx):
    """Parallel objective: optimize multiple features simultaneously.
    
    Suggests parameters for N features at once (ch_0, t_0, c_0, r_0, ..., ch_N-1, ...)
    and evaluates the combined feature set.
    """
    n_features = ctx.get('n_target_features', 10)  # Default fallback
    
    # Extract all features in parallel with different parameters for each
    features = []
    for i in range(n_features):
        # Suggest different parameters for each feature
        ch = trial.suggest_int(f'ch_{i}', 0, ctx['n_channels'] - 1)
        t_idx = trial.suggest_int(f't_{i}', 0, len(ctx['transform_names']) - 1)
        c = trial.suggest_float(f'c_{i}', 0, 1)
        r = trial.suggest_float(f'r_{i}', 0, 1)
        s, e = get_segment_indices(c, r, ctx['n_time'])
        segment = ctx['transformed'][t_idx, :, ch, s:e+1]
        feat = segment.mean(axis=1, keepdims=True).astype(np.float32)
        features.append(feat)
    
    # Combine all features
    combined_feat = np.hstack(features) if len(features) > 1 else features[0]
    
    # Store for extraction mode
    ctx['last_feature'] = combined_feat
    if ctx.get('extract_only'):
        return 0.0
    
    # Evaluate combined feature set
    current_X = ctx['current_X']
    X = np.hstack([current_X, combined_feat]) if current_X.size else combined_feat
    model, y, metric, cv_splits = ctx['model'], ctx['y'], ctx['metric'], ctx['cv_splits']
    total = sum(model.evaluate(X[tr], y[tr], X[va], y[va], metric) for tr, va in cv_splits)
    return total / len(cv_splits)