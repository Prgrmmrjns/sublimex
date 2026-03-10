"""LightGBM wrapper for SublimeX. Interface: evaluate(...), test(...), predict/predict_proba."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error


def _as_df(X):
    """Ensure X has feature names so fit/predict don't trigger sklearn warnings."""
    if hasattr(X, 'columns'):
        return X
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    n_cols = X.shape[1]
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_cols)])


def _score(y_true, pred, metric):
    if metric == 'auc':
        p = pred[:, 1] if pred.ndim == 2 and pred.shape[1] == 2 else pred
        return roc_auc_score(y_true, p) if pred.ndim == 2 else roc_auc_score(y_true, pred)
    if metric == 'accuracy':
        return accuracy_score(y_true, pred.argmax(axis=1) if pred.ndim == 2 else pred)
    if metric == 'rmse':
        return np.sqrt(mean_squared_error(y_true, pred))
    raise ValueError(metric)


class LightGBMModel:
    def __init__(self, task='classification', max_depth=3, n_estimators=100, early_stopping_rounds=10):
        
        self.task = task
        self.model = None
        self._cls = LGBMClassifier if task == 'classification' else LGBMRegressor
        self._params = dict(max_depth=max_depth, num_leaves=2**max_depth, verbosity=-1, force_row_wise=True)
        self._n_estimators = n_estimators
        self._early_stopping_rounds = early_stopping_rounds
        self._early_stopping_fn = early_stopping

    def evaluate(self, X_train, y_train, X_val, y_val, metric):
        X_train, X_val = _as_df(X_train), _as_df(X_val)
        m = self._cls(**self._params, n_estimators=self._n_estimators)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[self._early_stopping_fn(self._early_stopping_rounds, verbose=False)])
        pred = m.predict_proba(X_val) if metric == 'auc' and self.task == 'classification' else m.predict(X_val)
        return _score(y_val, pred, metric)

    def test(self, X_train, y_train, X_test, y_test, metric):
        X_train, X_test = _as_df(X_train), _as_df(X_test)
        self.model = self._cls(**self._params, n_estimators=self._n_estimators)
        self.model.fit(X_train, y_train)
        pred = self.model.predict_proba(X_test) if metric == 'auc' and self.task == 'classification' else self.model.predict(X_test)
        return _score(y_test, pred, metric)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
