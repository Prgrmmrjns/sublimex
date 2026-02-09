# SublimeX

**Supervised Bottom-Up Localized Multi-Representative Feature eXtraction**

[![PyPI version](https://badge.fury.io/py/sublimex.svg)](https://badge.fury.io/py/sublimex)
[![Python 3.11+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SublimeX is an interpretable feature extraction framework for time series and spatial data. It discovers a minimal set of task-specific features through Bayesian optimization, where each feature has explicit, human-readable semantics.

## Key Features

- **Minimal feature sets**: Typically 5-15 features vs. thousands from other methods
- **Full interpretability**: Each feature = statistic over optimized segment of transformed signal
- **Competitive performance**: Matches deep learning on many tasks
- **Modular design**: Custom transforms, objectives, and ML models

## Flowchart of SublimeX pipeline

![SublimeX Flowchart](flowchart.png)

## Installation

```bash
pip install sublimex
```

Or install from source:

```bash
git clone https://github.com/Prgrmmrjns/SublimeX.git
cd SublimeX
pip install -e .
```

## Quick Start

```python
import sublimex
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Your data: list of arrays/DataFrames, one per channel
# Each array has shape (n_samples, n_time_points)
X = [channel1_data, channel2_data, channel3_data]  # 3 channels
y = labels  # Binary or continuous targets

# Split data (indices then index each channel by them)
idx_train, idx_test = train_test_split(
    range(len(y)), test_size=0.2, stratify=y, random_state=42
)
X_train = [x.iloc[idx_train] for x in X]
X_test = [x.iloc[idx_test] for x in X]
y_train, y_test = y[idx_train], y[idx_test]

# Fit SublimeX
model = sublimex.SublimeX(metric='auc', n_trials=100, verbose=True)
train_features = model.fit_transform(X_train, y_train)
test_features = model.transform(X_test)

# Use with any classifier
clf = RandomForestClassifier()
clf.fit(train_features, y_train)
predictions = clf.predict(test_features)

# Interpret: each feature = mean over (channel, transform, segment)
for desc in model.get_feature_descriptions():
    print(desc)
```

## How It Works

SublimeX discovers discriminative features through a simple but effective process:

1. **Signal Transformation**: Apply multiple transforms to create different "views" of the data (raw, z-score normalized, derivative, FFT power spectrum)

2. **Segment Optimization**: Use Bayesian optimization (Optuna) to find segments that maximize downstream model performance

3. **Feature Extraction**: Compute statistics (e.g., mean) over discovered segments

4. **Iterative Discovery**: Repeat until adding new features no longer improves performance

Each discovered feature is fully interpretable:
> "Mean of z-score normalized signal in channel 2, positions 40-60"

## Configuration

### Basic Parameters

```python
import sublimex

model = sublimex.SublimeX(
    metric='auc',          # 'auc', 'accuracy', or 'rmse'
    n_trials=300,          # Optimization trials per feature
    max_features=None,     # Cap number of features (None = stop when no improvement)
    inner_cv=1,            # Internal CV folds (1 = single split)
    val_size=0.5,          # Validation size when inner_cv=1
    random_state=42,       # Reproducibility (CV + Optuna)
    verbose=True,          # Print progress
)
```

### Custom Transforms

```python
import sublimex

# Pass custom transforms dict
model = sublimex.SublimeX(transforms={'raw': lambda d: d.astype(np.float32), 'zscore': your_fn})
```

## Extending SublimeX

SublimeX is designed so you can plug in your own components without forking the repo.

| Component | How to extend |
|-----------|----------------|
| **Transforms** | Pass `transforms={'name': fn}`. Each fn receives shape `(..., n_time)` and returns same shape. |
| **Objectives** | Implement `(trial, ctx) -> float`: use `trial.suggest_*` for segment/params, read `ctx['transformed']`, `ctx['n_channels']`, `ctx['n_time']`, etc., set `ctx['last_feature']` when `ctx['extract_only']` is True, else return mean CV score. See `sublimex.objectives` module docstring for full `ctx` fields. |
| **Models** | Any object with `evaluate(X_train, y_train, X_val, y_val, metric) -> float` and `test(X_train, y_train, X_test, y_test, metric) -> float`. Optionally `predict(X)` / `predict_proba(X)` after fitting. See `sublimex.models` module docstring. |

For ablation or fixed-size feature sets, set `max_features=N` so discovery stops after N features even if the metric could still improve.

## Built-in Options

### Transforms

| Transform | Description | Use Case |
|-----------|-------------|----------|
| `raw` | Identity (original signal) | Amplitude differences |
| `zscore` | Z-score normalization | Shape differences |
| `derivative` | First-order gradient | Rate of change, transitions |
| `fft_power` | FFT power spectrum | Frequency content, periodicity |

Default objective is **mean over segment** (one statistic per feature). Custom objectives: pass `objective_fn=(trial, ctx) -> float`.

## Visualization

```python
import sublimex
import matplotlib.pyplot as plt

# Feature importance (use with your classifier's importances)
fig = sublimex.plot_feature_importance(importances, labels=model.get_feature_descriptions())
plt.show()

# Where the first feature's segment falls on the signal
fig = sublimex.plot_segment_on_signal(data[0], model.extracted_features[0], model.n_time)
plt.show()
```

## Saving and Loading

```python
import sublimex

# Save discovered features
model.save_features('features.json')

# Load and reuse
new_model = sublimex.SublimeX()
new_model.load_features('features.json')
features = new_model.transform(X_new)
```


## Complete Example: Synthetic Multi-Channel Time Series

```python
"""Complete example with synthetic data."""
import sublimex
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Generate synthetic multi-channel time series
np.random.seed(42)
n_samples, n_channels, n_bins = 500, 5, 200

X = []
y = np.random.randint(0, 2, n_samples)

for channel in range(n_channels):
    channel_data = []
    for i in range(n_samples):
        t = np.linspace(0, 4 * np.pi, n_bins)
        if y[i] == 1:
            # Class 1: Higher amplitude with distinctive peak
            signal = 2.0 * np.sin(t) + 1.5 * np.sin(2 * t)
            peak = n_bins // 2
            signal[peak-20:peak+20] += 1.5 * np.exp(-((np.arange(40) - 20) ** 2) / 50)
        else:
            # Class 0: Lower amplitude, more noise
            signal = 1.0 * np.sin(t) + 0.5 * np.sin(3 * t)
        signal += np.random.normal(0, 0.3, n_bins)
        channel_data.append(signal)
    
    X.append(pd.DataFrame(channel_data, columns=[f'bin_{j}' for j in range(n_bins)]))

# Split data
idx_train, idx_test = train_test_split(
    range(len(y)), test_size=0.2, stratify=y, random_state=42
)
X_train = [x.iloc[idx_train].astype(np.float32) for x in X]
X_test = [x.iloc[idx_test].astype(np.float32) for x in X]
y_train, y_test = y[idx_train], y[idx_test]

# Discover features with SublimeX
model = sublimex.SublimeX(metric='auc', n_trials=50, verbose=True)
train_features = model.fit_transform(X_train, y_train)
test_features = model.transform(X_test)

print(f"Discovered {train_features.shape[1]} features")

# Train classifier
clf = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
clf.fit(train_features, y_train)
auc = roc_auc_score(y_test, clf.predict_proba(test_features)[:, 1])
print(f"Test AUC: {auc:.4f}")

# Interpret features
print("\nDiscovered Features:")
for desc in model.get_feature_descriptions():
    print(f"  {desc}")

# Visualize
import matplotlib.pyplot as plt

fig = sublimex.plot_feature_importance(
    clf.feature_importances_, 
    model.get_feature_descriptions()
)
plt.show()
```

## Citation

If you use SublimeX in your research, please cite:

```bibtex
@software{sublimex2025,
  title={Supervised Bottom-Up Localized Multi-Representative Feature eXtraction},
  author={Wolber, J.C.},
  year={2026},
  url={https://github.com/Prgrmmrjns/SublimeX}
}
```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
