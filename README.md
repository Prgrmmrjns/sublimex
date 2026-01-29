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
from sublimex import SublimeX

# Your data: list of arrays/DataFrames, one per channel
# Each array has shape (n_samples, n_time_points)
X_train = [channel1_train, channel2_train]
X_test = [channel1_test, channel2_test]

# Fit SublimeX
model = SublimeX(metric='auc', n_trials=100, verbose=True)
model.fit(X_train, y_train)

# Transform to features
features_train = model.transform(X_train)
features_test = model.transform(X_test)

# Use with any classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(features_train, y_train)
predictions = clf.predict(features_test)
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
model = SublimeX(
    metric='auc',          # 'auc', 'accuracy', or 'rmse'
    n_trials=300,          # Optimization trials per feature
    inner_cv=1,            # Internal CV folds (1 = single split)
    val_size=0.5,          # Validation size when inner_cv=1
    verbose=True,          # Print progress
)
```

### Custom Transforms

```python
from sublimex import register_transform

# Register a custom transform
def hilbert_envelope(data):
    from scipy.signal import hilbert
    return np.abs(hilbert(data, axis=-1))

register_transform('hilbert', hilbert_envelope)

# Use specific transforms
model = SublimeX(
    transforms={
        'raw': lambda x: x,
        'hilbert': hilbert_envelope,
    }
)
```

### Custom Objective Functions

```python
from sublimex import create_custom_objective

# Create objective with custom aggregation
def rms(segment):
    """Root mean square."""
    return np.sqrt((segment ** 2).mean(axis=1, keepdims=True))

rms_objective = create_custom_objective(rms, 'rms')
model = SublimeX(objective_fn=rms_objective)
```

### Custom ML Models

```python
from sublimex import SklearnModelWrapper
from sklearn.ensemble import RandomForestClassifier

# Wrap any sklearn estimator
rf = RandomForestClassifier(n_estimators=100)
model = SublimeX(model=SklearnModelWrapper(rf))
```

## Built-in Options

### Transforms

| Transform | Description | Use Case |
|-----------|-------------|----------|
| `raw` | Identity (original signal) | Amplitude differences |
| `zscore` | Z-score normalization | Shape differences |
| `derivative` | First-order gradient | Rate of change, transitions |
| `fft_power` | FFT power spectrum | Frequency content, periodicity |

### Aggregations (for `aggregate_objective`)

| Aggregation | Description |
|-------------|-------------|
| `mean` | Average value (default) |
| `min`, `max` | Extreme values |
| `range` | max - min |
| `std` | Standard deviation |
| `median` | Robust central tendency |
| `argmin`, `argmax` | Position of extrema |

### Objectives

| Objective | Description |
|-----------|-------------|
| `mean_objective` | Mean over segment (default, most interpretable) |
| `aggregate_objective` | Choose from 8 aggregations |
| `pattern_objective` | B-spline pattern matching |

## Visualization

```python
from sublimex.visualization import (
    plot_feature_distributions,
    plot_segment_on_signal,
    plot_transform_comparison,
)

# Compare feature values across classes
fig = plot_feature_distributions(features, y, feature_idx=0)

# Show where a feature's segment falls on the signal
fig = plot_segment_on_signal(signal, model.extracted_features[0], model.n_time)

# Compare all transforms for a sample
fig = plot_transform_comparison(signal, TRANSFORMS)
```

## Saving and Loading

```python
# Save discovered features
model.save_features('features.json')

# Load and reuse
new_model = SublimeX()
new_model.load_features('features.json')
new_model.n_time = 200  # Set from original data
new_model.n_channels = 5
features = new_model.transform(X_new)
```

## Example: Gene Expression Prediction

```python
"""Predict gene expression from histone modifications (REMC dataset)."""
from sublimex import SublimeX

# Load histone mark signals (5 marks × 200 bins around TSS)
X = [h3k4me3_df, h3k4me1_df, h3k36me3_df, h3k9me3_df, h3k27me3_df]
y = expression_labels  # 0 = low, 1 = high

# Discover features
model = SublimeX(metric='auc', n_trials=100, verbose=True)
model.fit(X, y)

# Interpret
for desc in model.get_feature_descriptions():
    print(desc)
# Output:
# Feature 1: mean of raw in channel 0, positions 90-110
# Feature 2: mean of derivative in channel 2, positions 100-180
# ...
```

## Comparison with Other Methods

| Method | Features | Interpretability | Optimization |
|--------|----------|------------------|--------------|
| **SublimeX** | 5-15 | High (explicit segments) | Bayesian |
| tsfresh | 100-800 | Medium (statistical) | Filter |
| catch22 | 22 | Medium (fixed set) | None |
| MiniRocket | ~10,000 | Low | Deterministic |
| RDST | 2k-10k | Medium (shapelets) | Random |

## Citation

If you use SublimeX in your research, please cite:

```bibtex
@software{sublimex2025,
  title={SublimeX: Supervised Bottom-Up Localized Multi-Representative Feature eXtraction},
  author={Wolber, J.C.},
  year={2026},
  url={https://github.com/Prgrmmrjns/SublimeX}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
