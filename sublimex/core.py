"""SublimeX: interpretable feature extraction from multi-channel time series.

Input: single DataFrame/array -> univariate (1 channel); list of DataFrames/arrays -> multivariate.
1. Stack channels -> (n_samples, n_channels, n_time).
2. Apply transforms to get views (raw, zscore, fft_power, ...).
3. Until no improvement: Optuna picks (channel, transform, center, range); extract mean over segment; add feature if validation metric improves.
4. transform() applies saved segment/mean to new data.
"""
import json
import os
import numpy as np
import warnings
from typing import List, Dict, Any, Optional, Union
import pandas as pd

import optuna
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from sublimex.transforms import TRANSFORMS
from sublimex.objectives import mean_objective
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


def _to_channel_list(input_series: Union[Any, List]) -> List[np.ndarray]:
    """Normalize to list of (n_samples, n_time) float32 arrays.

    Univariate vs multivariate is detected as follows:
    - 2D array or DataFrame (n_samples, n_time) → univariate (1 channel).
    - 3D array (n_samples, n_channels, n_time) → multivariate; one channel per slice.
    - List of 2D arrays/DataFrames (one per channel) → multivariate.
    """
    if isinstance(input_series, (list, tuple)):
        # Multivariate: list of DataFrames or arrays, one per channel
        out = []
        for ch in input_series:
            arr = np.asarray(ch, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            out.append(arr)
        return out
    # Single object: DataFrame or array
    X = np.asarray(input_series, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim == 2:
        return [X]  # univariate (n_samples, n_time)
    if X.ndim == 3:
        # 3D array (n_samples, n_channels, n_time) → multivariate
        return [X[:, i, :] for i in range(X.shape[1])]
    raise ValueError("input_series must be 1D/2D/3D array or DataFrame, or list of 2D.")


def _segment_indices(center: float, range_val: float, n_time: int) -> tuple:
    """Segment boundaries (start, end) from normalized center and range."""
    center_idx = center * (n_time - 1)
    half = (range_val * (n_time - 1)) * 0.5
    return max(0, int(center_idx - half)), min(n_time - 1, int(center_idx + half))


class FeatureExtractor:
    """Optimize segment (channel, transform, time window) and mean aggregation per feature; metric improves on validation.

    After fit(), the downstream model is fitted and stored: use model = feature_extractor.model to get the wrapper
    (e.g. model.predict(feature_extractor.transform(X)) or model.model for the raw classifier).
    """

    def __init__(self, metric='accuracy', n_trials=300, max_features=None, inner_cv=1, val_size=0.5,
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
        self.objective_fn = objective_fn or mean_objective
        self.model = model
        self.extracted_features: List[Dict[str, Any]] = []
        self.transform_names: List[str] = list(self.transforms.keys())
        self.n_channels: Optional[int] = None
        self.n_time: Optional[int] = None
        self._is_fitted = False

    def _get_feature_name(self, params: Dict[str, Any]) -> str:
        """Human-readable name from params: e.g. raw_0-23_ch1 (channel omitted when univariate)."""
        t = params.get('t', 0)
        transform_name = self.transform_names[t] if t < len(self.transform_names) else f"t{t}"
        c, r = params.get('c', 0.5), params.get('r', 0.5)
        n_time = self.n_time or 1
        start, end = _segment_indices(c, r, n_time)
        name = f"{transform_name}_{start}-{end}"
        if self.n_channels is not None and self.n_channels > 1:
            name += f"_ch{params.get('ch', 0)}"
        return name

    def get_feature_names(self) -> List[str]:
        """Names of extracted features (e.g. raw_0-23_ch1). Empty if not fitted."""
        return [self._get_feature_name(p) for p in self.extracted_features]

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
        channels = _to_channel_list(input_series)
        data = np.stack(channels, axis=1).astype(np.float32)
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
        # Fit the downstream model on extracted features so model = sx.model is ready for predict
        self.model.test(current_X, y, current_X, y, self.metric)
        self._is_fitted = True
        return self

    def transform(self, input_series, initial_X=None):
        channels = _to_channel_list(input_series)
        data = np.stack(channels, axis=1).astype(np.float32)
        n_samples, n_channels, n_time = data.shape
        transformed = self._apply_transforms(data)
        ctx = {'transformed': transformed, 'n_time': n_time, 'n_channels': n_channels, 'transform_names': self.transform_names}
        feats = [self._extract_one(p, ctx) for p in self.extracted_features]
        out = np.hstack(feats).astype(np.float32)
        names = self.get_feature_names()
        if initial_X is not None:
            initial_X = np.asarray(initial_X, dtype=np.float32).reshape(n_samples, -1)
            out = np.hstack([initial_X, out])
            n_initial = initial_X.shape[1]
            names = [f"initial_{i}" for i in range(n_initial)] + names
        if pd is not None and len(names) == out.shape[1]:
            return pd.DataFrame(out, columns=names)
        return out

    def fit_transform(self, input_series, y, initial_X=None):
        return self.fit(input_series, y, initial_X=initial_X).transform(input_series, initial_X=initial_X)

    def save_features(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        out = [{k: (int(v) if isinstance(v, (np.integer, int)) else float(v) if isinstance(v, (np.floating, float)) else v) for k, v in p.items()} for p in self.extracted_features]
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)