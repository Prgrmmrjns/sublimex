"""ML model wrappers for SublimeX.

To use a custom model with SublimeX, your object must implement:

  - evaluate(X_train, y_train, X_val, y_val, metric) -> float
    Train on (X_train, y_train), predict on X_val, return the metric value.
  - test(X_train, y_train, X_test, y_test, metric) -> float
    Train on full train set, evaluate on test set (used for final evaluation).
  - predict(X) -> array (and optionally predict_proba(X) for classification)
    Used after test() to obtain predictions from the fitted model.

metric is one of 'auc', 'accuracy', 'rmse'. For 'auc', classifiers should
return predict_proba in evaluate/test so that AUC can be computed.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error


def _compute_score(y_true, pred, metric):
    """Compute evaluation metric."""
    if metric == 'auc':
        if pred.ndim == 2 and pred.shape[1] == 2:
            return roc_auc_score(y_true, pred[:, 1])
        elif pred.ndim == 2:
            return roc_auc_score(y_true, pred, multi_class='ovr')
        return roc_auc_score(y_true, pred)
    elif metric == 'accuracy':
        if pred.ndim == 2:
            pred = pred.argmax(axis=1)
        return accuracy_score(y_true, pred)
    elif metric == 'rmse':
        return np.sqrt(mean_squared_error(y_true, pred))
    raise ValueError(f"Unknown metric: {metric}")


class LightGBMModel:
    """LightGBM wrapper for SublimeX evaluation loop."""
    
    _BASE_PARAMS = {
        'max_depth': 3, 
        'data_sample_strategy': 'goss',
        'verbosity': -1,
        'force_row_wise': True,
    }
    
    def __init__(self, task='classification', max_depth=3, n_estimators=100, early_stopping_rounds=10):
        self.task = task
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        
        from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
        self._clf_cls = LGBMClassifier
        self._reg_cls = LGBMRegressor
        self._early_stopping = early_stopping
        self._model_cls = self._clf_cls if task == 'classification' else self._reg_cls
    
    def evaluate(self, X_train, y_train, X_val, y_val, metric):
        """Train and evaluate on validation set."""
        params = {**self._BASE_PARAMS, 'num_threads': 1}
        model = self._model_cls(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[self._early_stopping(self.early_stopping_rounds, verbose=False)])
        pred = model.predict_proba(X_val) if metric == 'auc' and self.task == 'classification' else model.predict(X_val)
        return _compute_score(y_val, pred, metric)
    
    def test(self, X_train, y_train, X_test, y_test, metric):
        """Train on full training set and evaluate on test set."""
        params = {**self._BASE_PARAMS, 'num_threads': -1}
        self.model = self._model_cls(**params)
        self.model.fit(X_train, y_train)
        pred = self.model.predict_proba(X_test) if metric == 'auc' and self.task == 'classification' else self.model.predict(X_test)
        return _compute_score(y_test, pred, metric)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)


class SklearnModelWrapper:
    """Wrapper for any sklearn estimator."""
    
    def __init__(self, estimator, clone_for_each_eval=True):
        self.estimator = estimator
        self.clone_for_each_eval = clone_for_each_eval
        self.model = None
        self._is_classifier = hasattr(estimator, 'predict_proba')
        from sklearn.base import clone
        self._clone = clone
    
    def _get_estimator(self):
        return self._clone(self.estimator) if self.clone_for_each_eval else self.estimator
    
    def evaluate(self, X_train, y_train, X_val, y_val, metric):
        """Train and evaluate on validation set."""
        model = self._get_estimator()
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val) if metric == 'auc' and self._is_classifier else model.predict(X_val)
        return _compute_score(y_val, pred, metric)
    
    def test(self, X_train, y_train, X_test, y_test, metric):
        """Train on full training set and evaluate on test set."""
        self.model = self._get_estimator()
        self.model.fit(X_train, y_train)
        pred = self.model.predict_proba(X_test) if metric == 'auc' and self._is_classifier else self.model.predict(X_test)
        return _compute_score(y_test, pred, metric)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
