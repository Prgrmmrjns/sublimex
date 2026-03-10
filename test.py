"""Synthetic multi-channel time series example: fit and predict with SublimeX."""
import sublimex as slx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Synthetic data: multi-channel time series with discriminative segment, high noise ---
np.random.seed(42)
n_samples, n_ch, n_time = 400, 5, 100
t = np.linspace(0, 4 * np.pi, n_time)
noise_scale = 1.2   # strong noise
signal_scale = 0.9  # weak discriminative bump (harder to find)

# Class 0: baseline + noise. Class 1: small bump in middle of channel 0 (discriminative).
X_3d = np.zeros((n_samples, n_ch, n_time), dtype=np.float32)
y = np.random.randint(0, 2, n_samples)
for i in range(n_samples):
    for ch in range(n_ch):
        X_3d[i, ch, :] = np.sin(t) + noise_scale * np.random.randn(n_time).astype(np.float32)
    if y[i] == 1:
        peak = n_time // 2
        X_3d[i, 0, peak - 12 : peak + 12] += signal_scale  # weak bump in channel 0

# --- Train / test split (do this before building the channel list) ---
train_idx, test_idx = train_test_split(range(n_samples), test_size=0.2, stratify=y, random_state=42)

# --- Multivariate input: list of DataFrames, one per channel ---
# SublimeX expects (a) list of DataFrames/arrays, each (n_samples, n_time), or (b) 3D array (n_samples, n_channels, n_time).
X = [pd.DataFrame(X_3d[:, ch, :]) for ch in range(n_ch)]
X_train = [x.iloc[train_idx] for x in X]
X_test = [x.iloc[test_idx] for x in X]
y_train, y_test = y[train_idx], y[test_idx]

# --- Fit feature extractor (extracts features and fits downstream model) ---
feature_extractor = slx.FeatureExtractor(metric="accuracy", verbose=True, n_trials=100, max_features=5)
feat_train = feature_extractor.fit_transform(X_train, y_train)
feat_test = feature_extractor.transform(X_test)

# --- Predict using the stored model (fitted at end of fit()) ---
model = feature_extractor.model
pred = model.predict(feat_test)
accuracy = accuracy_score(y_test, pred)
print(f"Extracted {feat_train.shape[1]} features, Test accuracy: {accuracy:.4f}")
