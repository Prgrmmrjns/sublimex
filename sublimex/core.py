"""SublimeX core module - interpretable feature extraction."""
import numpy as np
import json
import os
import warnings
from typing import List, Dict, Any, Optional

import optuna
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from sublimex.transforms import TRANSFORMS
from sublimex.objectives import default_objective
from sublimex.models import LightGBMModel

warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class MockTrial:
    """Mock Optuna trial for replaying saved parameters."""
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
    """SublimeX: Supervised Bottom-Up Localized Multi-Representative Feature Extraction."""
    
    def __init__(self, metric='auc', n_trials=300, inner_cv=1, val_size=0.5,
                 verbose=False, show_progress_bar=False, transforms=None,
                 objective_fn=None, model=None, sampler='tpe'):
        self.metric = metric
        self.n_trials = n_trials
        self.inner_cv = inner_cv
        self.val_size = val_size
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.transforms = transforms or TRANSFORMS
        self.objective_fn = objective_fn or default_objective
        self.model = model
        self.sampler = sampler
        
        self.extracted_features: List[Dict[str, Any]] = []
        self.transform_names: List[str] = list(self.transforms.keys())
        self.n_channels: Optional[int] = None
        self.n_time: Optional[int] = None
        self._is_fitted: bool = False
    
    def _apply_transforms(self, data):
        """Apply all transforms to input data."""
        n_samples, n_channels, n_time = data.shape
        out = np.empty((len(self.transform_names), n_samples, n_channels, n_time), dtype=np.float32)
        for ti, tname in enumerate(self.transform_names):
            flat = data.reshape(-1, n_time)
            out[ti] = self.transforms[tname](flat).reshape(n_samples, n_channels, n_time)
        return out
    
    def _extract_feature(self, params, ctx):
        """Extract a single feature using saved parameters."""
        ctx['extract_only'] = True
        self.objective_fn(MockTrial(params), ctx)
        ctx['extract_only'] = False
        return ctx['last_feature']
    
    def _create_cv_splits(self, n_samples, y):
        """Create CV splits for internal evaluation."""
        if self.inner_cv == 1:
            stratify = y if self.metric != 'rmse' else None
            train_idx, val_idx = train_test_split(
                np.arange(n_samples), test_size=self.val_size, 
                random_state=42, stratify=stratify)
            return [(train_idx, val_idx)]
        cv_cls = StratifiedKFold if self.metric != 'rmse' else KFold
        return list(cv_cls(self.inner_cv, shuffle=True, random_state=42).split(np.zeros(n_samples), y))
    
    def _create_sampler(self):
        """Create Optuna sampler."""
        if self.sampler == 'nsga2':
            return optuna.samplers.NSGAIISampler()
        return optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True)
    
    def fit(self, input_series, y):
        """Fit the feature extractor to training data."""
        data = np.stack(list(input_series), axis=1).astype(np.float32)
        n_samples, self.n_channels, self.n_time = data.shape
        
        if self.verbose:
            print(f"\nSublimeX Feature Extraction")
            print(f"  Samples: {n_samples}")
            print(f"  Channels: {self.n_channels}")
            print(f"  Time points: {self.n_time}")
            print(f"  Transforms: {self.transform_names}")
            print(f"  Metric: {self.metric}\n")
        
        if self.model is None:
            task = 'regression' if self.metric == 'rmse' else 'classification'
            self.model = LightGBMModel(task=task)
        
        cv_splits = self._create_cv_splits(n_samples, y)
        transformed = self._apply_transforms(data)
        
        direction = 'minimize' if self.metric == 'rmse' else 'maximize'
        is_maximize = direction == 'maximize'
        
        ctx = {
            'transformed': transformed, 'y': y, 'model': self.model,
            'metric': self.metric, 'n_channels': self.n_channels,
            'n_time': self.n_time, 'transform_names': self.transform_names,
            'cv_splits': cv_splits,
        }
        
        self.extracted_features = []
        current_X = np.empty((n_samples, 0), dtype=np.float32)
        best_score = float('inf') if not is_maximize else -float('inf')
        while True:
            ctx['current_X'] = current_X
            
            study = optuna.create_study(direction=direction, sampler=self._create_sampler())
            study.optimize(lambda t: self.objective_fn(t, ctx), n_trials=self.n_trials,
                          show_progress_bar=self.show_progress_bar, n_jobs=-1)
            
            improved = (is_maximize and study.best_value > best_score) or \
                       (not is_maximize and study.best_value < best_score)
            
            if not improved:
                break
            
            best_score = study.best_value
            params = study.best_params
            self.extracted_features.append(params)
            
            feat = self._extract_feature(params, ctx)
            current_X = np.hstack([current_X, feat]) if current_X.size else feat
            
            if self.verbose:
                print(f"  Feature {len(self.extracted_features)}: {self.metric}={best_score:.5f}, params={params}")
        
        self._is_fitted = True
        if self.verbose:
            print(f"\nDiscovered {len(self.extracted_features)} features")
        return self
    
    def transform(self, input_series):
        """Transform data using extracted features."""
        
        data = np.stack(list(input_series), axis=1).astype(np.float32)
        n_samples, n_channels, n_time = data.shape
        transformed = self._apply_transforms(data)
        
        ctx = {'transformed': transformed, 'n_time': n_time, 
               'n_channels': n_channels, 'transform_names': self.transform_names}
        
        features = [self._extract_feature(p, ctx) for p in self.extracted_features]
        return np.hstack(features).astype(np.float32)
    
    def fit_transform(self, input_series, y):
        """Fit and transform in one step."""
        return self.fit(input_series, y).transform(input_series)
    
    def save_features(self, path):
        """Save extracted feature parameters to JSON file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        features_list = []
        for i, params in enumerate(self.extracted_features):
            feature_dict = {'feature_id': i + 1}
            for k, v in params.items():
                if isinstance(v, (np.integer, int)):
                    feature_dict[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    feature_dict[k] = float(v)
                else:
                    feature_dict[k] = v
            features_list.append(feature_dict)
        with open(path, 'w') as f:
            json.dump(features_list, f, indent=2)
    
    def load_features(self, path):
        """Load feature parameters from JSON file."""
        with open(path, 'r') as f:
            features_list = json.load(f)
        self.extracted_features = [{k: v for k, v in f.items() if k != 'feature_id'} 
                                   for f in features_list]
        self._is_fitted = True
        return self
    
    def get_feature_descriptions(self):
        """Get human-readable descriptions of extracted features."""
        if not self.extracted_features:
            return []
        
        descriptions = []
        for i, params in enumerate(self.extracted_features):
            t_idx = params.get('t', 0)
            transform = self.transform_names[t_idx] if t_idx < len(self.transform_names) else f"transform_{t_idx}"
            ch = params.get('ch', 0)
            c, r = params.get('c', 0.5), params.get('r', 0.5)
            
            if self.n_time:
                start = int(c * self.n_time - r * self.n_time / 2)
                end = int(c * self.n_time + r * self.n_time / 2)
                pos_str = f"positions {max(0, start)}-{min(self.n_time, end)}"
            else:
                pos_str = f"center={c:.2f}, range={r:.2f}"
            
            feat_type = params.get('feature_type', 'mean')
            descriptions.append(f"Feature {i+1}: {feat_type} of {transform} in channel {ch}, {pos_str}")
        return descriptions
    
    def __repr__(self):
        status = "fitted" if self._is_fitted else "not fitted"
        return f"SublimeX(metric='{self.metric}', n_trials={self.n_trials}, status={status}, n_features={len(self.extracted_features)})"
