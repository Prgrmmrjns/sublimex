"""SublimeX: interpretable feature extraction from multi-channel time series.

1. Stack channels -> (n_samples, n_channels, n_time).
2. Apply transforms to get views (raw, zscore, fft_power, ...).
3. Until no improvement: Optuna picks (channel, transform, center, range); extract mean over segment; add feature if validation metric improves.
4. transform() applies saved segment/mean to new data.
"""
import json
import os
import numpy as np
import warnings
from typing import List, Dict, Any, Optional

import optuna
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from sublimex.transforms import TRANSFORMS
from sublimex.objectives import default_objective
from sublimex.models import LightGBMModel

warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class _MockTrial:
    __slots__ = ('params',)

    def __init__(self, params):
        self.params = params

    def suggest_int(self, name, low, high):
        return int(self.params[name])

    def suggest_float(self, name, low, high):
        return float(self.params[name])

    def suggest_categorical(self, name, choices):
        return self.params[name]


class SublimeX:
    """Optimize segment (channel, transform, time window) and mean aggregation per feature; metric improves on validation."""

    def __init__(self, metric='auc', n_trials=300, max_features=None, inner_cv=1, val_size=0.5,
                 random_state=None, verbose=False, show_progress_bar=False, transforms=None,
                 objective_fn=None, model=None):
        self.metric = metric
        self.n_trials = n_trials
        self.max_features = max_features
        self.inner_cv = inner_cv
        self.val_size = val_size
        self.random_state = random_state
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.transforms = transforms or TRANSFORMS
        self.objective_fn = objective_fn or default_objective
        self.model = model
        self.extracted_features: List[Dict[str, Any]] = []
        self.transform_names: List[str] = list(self.transforms.keys())
        self.n_channels: Optional[int] = None
        self.n_time: Optional[int] = None
        self._is_fitted = False

    def _apply_transforms(self, data):
        n_samples, n_channels, n_time = data.shape
        out = np.empty((len(self.transform_names), n_samples, n_channels, n_time), dtype=np.float32)
        for ti, tname in enumerate(self.transform_names):
            flat = data.reshape(-1, n_time)
            out[ti] = self.transforms[tname](flat).reshape(n_samples, n_channels, n_time)
        return out

    def _extract_one(self, params, ctx):
        ctx['extract_only'] = True
        self.objective_fn(_MockTrial(params), ctx)
        ctx['extract_only'] = False
        return ctx['last_feature']

    def _cv_splits(self, n_samples, y):
        if self.inner_cv == 1:
            stratify = y if self.metric != 'rmse' else None
            tr, va = train_test_split(np.arange(n_samples), test_size=self.val_size, random_state=self.random_state, stratify=stratify)
            return [(tr, va)]
        cls = StratifiedKFold if self.metric != 'rmse' else KFold
        return list(cls(self.inner_cv, shuffle=True, random_state=self.random_state).split(np.zeros(n_samples), y))

    def fit(self, input_series, y, initial_X=None):
        data = np.stack(list(input_series), axis=1).astype(np.float32)
        n_samples, self.n_channels, self.n_time = data.shape
        if self.verbose:
            print(f"SublimeX: {n_samples} samples, {self.n_channels} ch, {self.n_time} time, metric={self.metric}")
        if self.model is None:
            self.model = LightGBMModel(task='regression' if self.metric == 'rmse' else 'classification')
        cv_splits = self._cv_splits(n_samples, y)
        transformed = self._apply_transforms(data)
        direction = 'minimize' if self.metric == 'rmse' else 'maximize'
        is_max = direction == 'maximize'
        ctx = {'transformed': transformed, 'y': y, 'model': self.model, 'metric': self.metric,
               'n_channels': self.n_channels, 'n_time': self.n_time, 'transform_names': self.transform_names, 'cv_splits': cv_splits}
        self.extracted_features = []
        current_X = np.asarray(initial_X, dtype=np.float32).reshape(n_samples, -1) if initial_X is not None else np.empty((n_samples, 0), dtype=np.float32)
        best = -np.inf if is_max else np.inf
        while True:
            if self.max_features is not None and len(self.extracted_features) >= self.max_features:
                break
            ctx['current_X'] = current_X
            study = optuna.create_study(direction=direction)
            study.optimize(lambda t: self.objective_fn(t, ctx), n_trials=self.n_trials, show_progress_bar=self.show_progress_bar, n_jobs=-1)
            improved = (is_max and study.best_value > best) or (not is_max and study.best_value < best)
            if not improved:
                break
            best = study.best_value
            params = study.best_params
            self.extracted_features.append(params)
            feat = self._extract_one(params, ctx)
            current_X = np.hstack([current_X, feat]) if current_X.size else feat
            if self.verbose:
                print(f"  Feature {len(self.extracted_features)}: {self.metric}={best:.5f}")
        self._is_fitted = True
        return self

    def transform(self, input_series, initial_X=None):
        data = np.stack(list(input_series), axis=1).astype(np.float32)
        n_samples, n_channels, n_time = data.shape
        transformed = self._apply_transforms(data)
        ctx = {'transformed': transformed, 'n_time': n_time, 'n_channels': n_channels, 'transform_names': self.transform_names}
        feats = [self._extract_one(p, ctx) for p in self.extracted_features]
        out = np.hstack(feats).astype(np.float32)
        if initial_X is not None:
            initial_X = np.asarray(initial_X, dtype=np.float32).reshape(n_samples, -1)
            return np.hstack([initial_X, out])
        return out

    def fit_transform(self, input_series, y, initial_X=None):
        return self.fit(input_series, y, initial_X=initial_X).transform(input_series, initial_X=initial_X)

    def save_features(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        out = [{k: (int(v) if isinstance(v, (np.integer, int)) else float(v) if isinstance(v, (np.floating, float)) else v) for k, v in p.items()} for p in self.extracted_features]
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)