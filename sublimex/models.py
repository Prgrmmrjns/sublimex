"""LightGBM wrapper for manuscript experiments."""
import warnings
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

warnings.filterwarnings('ignore', message='X does not have valid feature names')
BASE_PARAMS = {'max_depth': 5, 'verbosity': -1, 'force_row_wise': True}


def encode_labels(y_train, y_test):
    labels = np.unique(np.concatenate([y_train, y_test]))
    label_map = {l: i for i, l in enumerate(labels)}
    return np.array([label_map[y] for y in y_train]), np.array([label_map[y] for y in y_test]), len(labels)


def _score(y_true, pred, metric):
    if metric == 'auc':
        return roc_auc_score(y_true, pred[:, 1] if pred.ndim == 2 else pred)
    if metric == 'rmse':
        return np.sqrt(mean_squared_error(y_true, pred))
    return accuracy_score(y_true, pred)


class LightGBMModel:
    def __init__(self, task='classification'):
        self.task = task
        self._cls = LGBMClassifier if task == 'classification' else LGBMRegressor

    def evaluate(self, X_train, y_train, X_val, y_val, metric):
        model = self._cls(**BASE_PARAMS, num_threads=1)
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val) if metric == 'auc' else model.predict(X_val)
        return _score(y_val, pred, metric)

    def test(self, X_train, y_train, X_test, y_test, metric, sample_weight=None, deterministic=False):
        params = {**BASE_PARAMS, 'num_threads': -1}
        if deterministic:
            params['random_state'] = 42
        model = self._cls(**params)
        fit_kwargs = {'sample_weight': sample_weight} if sample_weight is not None else {}
        model.fit(X_train, y_train, **fit_kwargs)
        pred = model.predict_proba(X_test) if metric == 'auc' else model.predict(X_test)
        return _score(y_test, pred, metric)
