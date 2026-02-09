"""Objective: mean over segment (default). Signature (trial, ctx) -> float."""
import numpy as np
from typing import Dict, Any


def _segment_indices(center: float, range_val: float, n_time: int) -> tuple:
    center_idx = center * (n_time - 1)
    half = (range_val * (n_time - 1)) * 0.5
    return max(0, int(center_idx - half)), min(n_time - 1, int(center_idx + half))


def _evaluate(feature: np.ndarray, ctx: Dict[str, Any]) -> float:
    ctx['last_feature'] = feature
    if ctx.get('extract_only'):
        return 0.0
    X = np.hstack([ctx['current_X'], feature]) if ctx['current_X'].size else feature
    scores = [ctx['model'].evaluate(X[tr], ctx['y'][tr], X[va], ctx['y'][va], ctx['metric'])
              for tr, va in ctx['cv_splits']]
    return np.mean(scores)


def mean_objective(trial, ctx: Dict[str, Any]) -> float:
    """Suggest channel, transform, center, range; extract mean over segment; return CV score."""
    ch = trial.suggest_int('ch', 0, ctx['n_channels'] - 1)
    t = trial.suggest_int('t', 0, len(ctx['transform_names']) - 1)
    c, r = trial.suggest_float('c', 0, 1), trial.suggest_float('r', 0, 1)
    start, end = _segment_indices(c, r, ctx['n_time'])
    segment = ctx['transformed'][t, :, ch, start:end + 1]
    feature = segment.mean(axis=1, keepdims=True).astype(np.float32)
    return _evaluate(feature, ctx)
