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
n_feat = 5
model = sublimex.SublimeX(metric='auc', n_trials=80, max_features=n_feat, verbose=True)
train_features = model.fit_transform(X_train, y_train)
test_features = model.transform(X_test)

n = train_features.shape[1]
print(f"Discovered {n} feature{'s' if n != 1 else ''}")

# Train classifier
clf = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
clf.fit(train_features, y_train)
auc = roc_auc_score(y_test, clf.predict_proba(test_features)[:, 1])
print(f"Test AUC: {auc:.4f}")
