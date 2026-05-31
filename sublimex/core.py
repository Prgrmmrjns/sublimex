"""SublimeX: sequential feature extraction (SublimeX + FeatureExtractor APIs)."""
import json
import os
import warnings
import numpy as np
import pandas as pd
import optuna
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sublimex.models import LightGBMModel
from sublimex.transforms import TRANSFORMS as PKG_TRANSFORMS
from sublimex.objectives import mean_objective as pkg_mean_objective

warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings('ignore', message='overflow encountered in reduce')
optuna.logging.set_verbosity(optuna.logging.WARNING)

VAL_SIZE = 0.5
RANDOM_STATE = 42

TRANSFORMS = {
    'raw': lambda d: d,
    'zscore': lambda d: (d - d.mean(axis=-1, keepdims=True)) / (d.std(axis=-1, keepdims=True) + 1e-8),
    'derivative': lambda d: np.gradient(d, axis=-1),
    'fft': lambda d: np.abs(np.fft.fft(d, axis=-1)),
}


def _get_segment(params, ctx):
    ch, t = int(params['ch']), int(params['t'])
    c, r, n = params['c'], params['r'], ctx['n_time']
    half = (r * (n - 1)) * 0.5
    s = max(0, int(c * (n - 1) - half))
    e = min(n - 1, int(c * (n - 1) + half))
    return ctx['transformed'][t, :, ch, s:e + 1]


AGGREGATIONS = {
    'mean': lambda x: x.mean(1, keepdims=True), 'std': lambda x: x.std(1, keepdims=True),
    'min': lambda x: x.min(1, keepdims=True), 'max': lambda x: x.max(1, keepdims=True),
    'range': lambda x: np.ptp(x, 1, keepdims=True), 'median': lambda x: np.median(x, 1, keepdims=True),
    'argmin': lambda x: x.argmin(1, keepdims=True).astype(np.float32) / max(x.shape[1] - 1, 1),
    'argmax': lambda x: x.argmax(1, keepdims=True).astype(np.float32) / max(x.shape[1] - 1, 1),
}


def suggest_segment_params(trial, ctx):
    return {
        'ch': trial.suggest_int('ch', 0, ctx['n_channels'] - 1),
        't': trial.suggest_int('t', 0, len(ctx['transform_names']) - 1),
        'c': trial.suggest_float('c', 0, 1),
        'r': trial.suggest_float('r', 0, 1),
    }


def extract_feature_from_params(params, ctx):
    seg = _get_segment(params, ctx)
    if 'w' in params:
        w = min(max(2, int(params['w'] * ctx['n_time'])), seg.shape[1])
        t = np.linspace(0, 1, w, dtype=np.float32)
        pat = ((1 - t)**2 * params['cp0'] + 2 * (1 - t) * t * params['cp1'] + t**2 * params['cp2']).astype(np.float32)
        d = sliding_window_view(seg.astype(np.float32), w, axis=1) - pat
        return (np.linalg.norm(d, axis=-1) / np.sqrt(w)).min(1, keepdims=True).astype(np.float32)
    if 'agg' in params:
        return AGGREGATIONS[params['agg']](seg).astype(np.float32)
    return seg.mean(1, keepdims=True).astype(np.float32)


def extract_feature(params, ctx):
    return extract_feature_from_params(params, ctx)


def _get_train_val_split(y, metric):
    stratify = y if metric != 'rmse' else None
    return train_test_split(np.arange(len(y)), test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=stratify)


def _evaluate(feat, ctx):
    X = np.hstack([ctx['current_X'], feat]) if ctx['current_X'].size else feat
    tr, va = _get_train_val_split(ctx['y'], ctx['metric'])
    return ctx['model'].evaluate(X[tr], ctx['y'][tr], X[va], ctx['y'][va], ctx['metric'])


def _suggest_segment(trial, ctx):
    return _get_segment(suggest_segment_params(trial, ctx), ctx)


def mean_objective(trial, ctx):
    return _evaluate(extract_feature_from_params(suggest_segment_params(trial, ctx), ctx), ctx)


def random_split_mean_objective(trial, ctx):
    feat = extract_feature_from_params(suggest_segment_params(trial, ctx), ctx)
    X = np.hstack([ctx['current_X'], feat]) if ctx['current_X'].size else feat
    stratify = ctx['y'] if ctx['metric'] != 'rmse' else None
    tr, va = train_test_split(np.arange(len(ctx['y'])), test_size=VAL_SIZE, stratify=stratify)
    return ctx['model'].evaluate(X[tr], ctx['y'][tr], X[va], ctx['y'][va], ctx['metric'])


def aggregate_objective(trial, ctx):
    p = suggest_segment_params(trial, ctx)
    p['agg'] = trial.suggest_categorical('agg', list(AGGREGATIONS))
    return _evaluate(extract_feature_from_params(p, ctx), ctx)


def pattern_objective(trial, ctx):
    p = suggest_segment_params(trial, ctx)
    p['w'] = trial.suggest_float('w', 0.05, 0.5)
    p['cp0'], p['cp1'], p['cp2'] = trial.suggest_float('cp0', 0, 1), trial.suggest_float('cp1', 0, 1), trial.suggest_float('cp2', 0, 1)
    return _evaluate(extract_feature_from_params(p, ctx), ctx)


def decision_tree_objective(trial, ctx):
    feat = extract_feature_from_params(suggest_segment_params(trial, ctx), ctx)
    X = np.hstack([ctx['current_X'], feat]) if ctx['current_X'].size else feat
    tr, va = _get_train_val_split(ctx['y'], ctx['metric'])
    Cls = DecisionTreeRegressor if ctx['metric'] == 'rmse' else DecisionTreeClassifier
    m = Cls(max_depth=5, random_state=RANDOM_STATE).fit(X[tr], ctx['y'][tr])
    pred = m.predict(X[va]) if ctx['metric'] == 'rmse' else m.predict_proba(X[va])
    if ctx['metric'] == 'rmse':
        return np.sqrt(mean_squared_error(ctx['y'][va], pred))
    return roc_auc_score(ctx['y'][va], pred[:, 1]) if ctx['metric'] == 'auc' else accuracy_score(ctx['y'][va], m.predict(X[va]))


def _parallel_objective(trial, ctx, n_parallel):
    feats = []
    for i in range(n_parallel):
        p = {'ch': trial.suggest_int(f'ch_{i}', 0, ctx['n_channels'] - 1),
             't': trial.suggest_int(f't_{i}', 0, len(ctx['transform_names']) - 1),
             'c': trial.suggest_float(f'c_{i}', 0, 1), 'r': trial.suggest_float(f'r_{i}', 0, 1)}
        feats.append(_get_segment(p, ctx).mean(axis=1, keepdims=True).astype(np.float32))
    return _evaluate(np.hstack(feats), ctx)


class SublimeX:
    def __init__(self, metric='auc', n_trials=300, n_workers=1, verbose=True, show_progress_bar=False, transforms=None,
                 objective_fn=None, extract_fn=None, sampler='tpe', deterministic=False):
        self.metric = metric
        self.n_trials = n_trials
        self.n_workers = n_workers
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.transforms = transforms or TRANSFORMS
        self.objective_fn = objective_fn
        self.extract_fn = extract_fn or extract_feature
        self.sampler = sampler
        self.deterministic = deterministic
        self.extracted_features = []
        self.transform_names = list(self.transforms.keys())
        self.n_channels = None
        self.n_time = None

    def _apply_transforms(self, data):
        n_samples, n_channels, n_time = data.shape
        out = np.empty((len(self.transform_names), n_samples, n_channels, n_time), dtype=np.float32)
        for ti, tname in enumerate(self.transform_names):
            out[ti] = self.transforms[tname](data.reshape(-1, n_time)).reshape(n_samples, n_channels, n_time)
        return out

    def _to_array(self, input_series):
        if isinstance(input_series, np.ndarray) and input_series.ndim == 3:
            return input_series.astype(np.float32, copy=False)
        if isinstance(input_series, list) and input_series and isinstance(input_series[0], np.ndarray):
            return np.stack([np.asarray(a, dtype=np.float32) for a in input_series], axis=1)
        arrays = [s.values.astype(np.float32) if hasattr(s, 'values') else np.asarray(s, dtype=np.float32) for s in input_series]
        return np.stack(arrays, axis=1).astype(np.float32)

    def _sampler(self):
        seed = RANDOM_STATE if self.deterministic else None
        if self.sampler == 'nsga2':
            return optuna.samplers.NSGAIISampler(seed=seed)
        return optuna.samplers.TPESampler(multivariate=True, constant_liar=not self.deterministic, seed=seed)

    def _optuna_jobs(self):
        return 1 if self.deterministic else self.n_workers

    def fit(self, input_series, y, initial_X=None, transformed=None):
        obj = self.objective_fn or mean_objective
        data = self._to_array(input_series)
        n_samples, self.n_channels, self.n_time = data.shape
        transformed = transformed if transformed is not None else self._apply_transforms(data)
        model = LightGBMModel('regression' if self.metric == 'rmse' else 'classification')
        direction = 'minimize' if self.metric == 'rmse' else 'maximize'
        ctx = {'transformed': transformed, 'y': y, 'model': model, 'metric': self.metric,
               'n_channels': self.n_channels, 'n_time': self.n_time,
               'transform_names': self.transform_names}
        self.extracted_features = []
        self._initial_X = np.asarray(initial_X, dtype=np.float32) if initial_X is not None else None
        current_X = self._initial_X.copy() if self._initial_X is not None else np.empty((n_samples, 0), dtype=np.float32)
        best_score = float('inf') if direction == 'minimize' else -float('inf')
        is_maximize = direction == 'maximize'
        while True:
            ctx['current_X'] = current_X
            study = optuna.create_study(direction=direction, sampler=self._sampler())
            study.optimize(lambda t: obj(t, ctx), n_trials=self.n_trials, show_progress_bar=self.show_progress_bar, n_jobs=self._optuna_jobs())
            improved = (is_maximize and study.best_value > best_score) or (not is_maximize and study.best_value < best_score)
            if not improved:
                break
            best_score = study.best_value
            self.extracted_features.append(study.best_params)
            feat = self.extract_fn(study.best_params, ctx)
            current_X = np.hstack([current_X, feat]) if current_X.size else feat
            if self.verbose:
                print(f'    + feature {len(self.extracted_features)}: val={best_score:.6f}', flush=True)
        return self

    def fit_parallel(self, input_series, y, n_parallel, initial_X=None, transformed=None):
        data = self._to_array(input_series)
        n_samples, self.n_channels, self.n_time = data.shape
        transformed = transformed if transformed is not None else self._apply_transforms(data)
        direction = 'minimize' if self.metric == 'rmse' else 'maximize'
        ctx = {'transformed': transformed, 'y': y,
               'model': LightGBMModel('regression' if self.metric == 'rmse' else 'classification'),
               'metric': self.metric, 'n_channels': self.n_channels, 'n_time': self.n_time,
               'transform_names': self.transform_names,
               'current_X': np.asarray(initial_X, dtype=np.float32) if initial_X is not None else np.empty((n_samples, 0), np.float32)}
        study = optuna.create_study(direction=direction, sampler=self._sampler())
        study.optimize(lambda t: _parallel_objective(t, ctx, n_parallel), n_trials=self.n_trials, n_jobs=self._optuna_jobs())
        bp = study.best_params
        self.extracted_features = [{'ch': int(bp[f'ch_{i}']), 't': int(bp[f't_{i}']), 'c': bp[f'c_{i}'], 'r': bp[f'r_{i}']} for i in range(n_parallel)]
        self._initial_X = np.asarray(initial_X, dtype=np.float32) if initial_X is not None else None
        return self

    def transform(self, input_series, initial_X=None, transformed=None):
        data = self._to_array(input_series)
        _, n_ch, n_time = data.shape
        self.n_channels, self.n_time = n_ch, n_time
        transformed = transformed if transformed is not None else self._apply_transforms(data)
        ctx = {'transformed': transformed, 'n_time': n_time, 'n_channels': n_ch, 'transform_names': self.transform_names}
        out = np.hstack([self.extract_fn(p, ctx) for p in self.extracted_features]).astype(np.float32)
        if initial_X is not None:
            out = np.hstack([np.asarray(initial_X, dtype=np.float32), out])
        elif getattr(self, '_initial_X', None) is not None:
            out = np.hstack([self._initial_X, out])
        return out

    def save_features(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump([{'feature_id': i + 1, **{k: float(v) if isinstance(v, (int, float, np.number)) else v
                        for k, v in p.items()}} for i, p in enumerate(self.extracted_features)], f, indent=2)

    def load_features(self, path):
        with open(path) as f:
            rows = json.load(f)
        self.extracted_features = [{k: v for k, v in row.items() if k != 'feature_id'} for row in rows]
        return self


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


def _to_channel_list(input_series):
    if isinstance(input_series, (list, tuple)):
        out = []
        for ch in input_series:
            arr = np.asarray(ch, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            out.append(arr)
        return out
    X = np.asarray(input_series, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim == 2:
        return [X]
    if X.ndim == 3:
        return [X[:, i, :] for i in range(X.shape[1])]
    raise ValueError("input_series must be 1D/2D/3D array or DataFrame, or list of 2D.")


def _segment_indices(center, range_val, n_time):
    center_idx = center * (n_time - 1)
    half = (range_val * (n_time - 1)) * 0.5
    return max(0, int(center_idx - half)), min(n_time - 1, int(center_idx + half))


class FeatureExtractor:
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
        self.transforms = transforms or PKG_TRANSFORMS
        self.objective_fn = objective_fn or pkg_mean_objective
        self.model = model
        self.extracted_features = []
        self.transform_names = list(self.transforms.keys())
        self.n_channels = None
        self.n_time = None

    def _get_feature_name(self, params):
        t = params.get('t', 0)
        transform_name = self.transform_names[t] if t < len(self.transform_names) else f"t{t}"
        start, end = _segment_indices(params.get('c', 0.5), params.get('r', 0.5), self.n_time or 1)
        name = f"{transform_name}_{start}-{end}"
        if self.n_channels and self.n_channels > 1:
            name += f"_ch{params.get('ch', 0)}"
        return name

    def get_feature_names(self):
        return [self._get_feature_name(p) for p in self.extracted_features]

    def _apply_transforms(self, data):
        n_samples, n_channels, n_time = data.shape
        out = np.empty((len(self.transform_names), n_samples, n_channels, n_time), dtype=np.float32)
        for ti, tname in enumerate(self.transform_names):
            out[ti] = self.transforms[tname](data.reshape(-1, n_time)).reshape(n_samples, n_channels, n_time)
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
        if self.model is None:
            self.model = LightGBMModel('regression' if self.metric == 'rmse' else 'classification')
        ctx = {'transformed': self._apply_transforms(data), 'y': y, 'model': self.model, 'metric': self.metric,
               'n_channels': self.n_channels, 'n_time': self.n_time, 'transform_names': self.transform_names,
               'cv_splits': self._cv_splits(n_samples, y)}
        self.extracted_features = []
        current_X = np.asarray(initial_X, dtype=np.float32).reshape(n_samples, -1) if initial_X is not None else np.empty((n_samples, 0), np.float32)
        direction = 'minimize' if self.metric == 'rmse' else 'maximize'
        is_max = direction == 'maximize'
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
            self.extracted_features.append(study.best_params)
            feat = self._extract_one(study.best_params, ctx)
            current_X = np.hstack([current_X, feat]) if current_X.size else feat
        self.model.test(current_X, y, current_X, y, self.metric)
        return self

    def transform(self, input_series, initial_X=None):
        channels = _to_channel_list(input_series)
        data = np.stack(channels, axis=1).astype(np.float32)
        n_samples = data.shape[0]
        ctx = {'transformed': self._apply_transforms(data), 'n_time': data.shape[2], 'n_channels': data.shape[1],
               'transform_names': self.transform_names}
        out = np.hstack([self._extract_one(p, ctx) for p in self.extracted_features]).astype(np.float32)
        names = self.get_feature_names()
        if initial_X is not None:
            initial_X = np.asarray(initial_X, dtype=np.float32).reshape(n_samples, -1)
            out = np.hstack([initial_X, out])
            names = [f"initial_{i}" for i in range(initial_X.shape[1])] + names
        return pd.DataFrame(out, columns=names) if len(names) == out.shape[1] else out

    def fit_transform(self, input_series, y, initial_X=None):
        return self.fit(input_series, y, initial_X=initial_X).transform(input_series, initial_X=initial_X)

    def save_features(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        out = [{k: (int(v) if isinstance(v, (np.integer, int)) else float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in p.items()} for p in self.extracted_features]
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
