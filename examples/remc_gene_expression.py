#!/usr/bin/env python3
"""REMC Gene Expression Prediction Example using SublimeX."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sublimex import SublimeX
from sublimex.models import LightGBMModel

DATA_PATH = '../../processed_datasets/remc/E003.parquet'
HISTONE_MARKS = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
N_TRIALS = 100
N_FOLDS = 5
RANDOM_STATE = 42


def load_remc_data(data_path=DATA_PATH):
    """Load REMC ChIP-seq data."""
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}\nGenerating synthetic data...")
        return generate_synthetic_data()
    
    df = pd.read_parquet(data_path)
    X = [df[[c for c in df.columns if c.startswith(f"{mark}_")]].copy() 
         for mark in HISTONE_MARKS]
    y = df['target'].values
    
    print(f"Loaded: {len(y)} samples, {len(HISTONE_MARKS)} channels, {X[0].shape[1]} bins")
    return X, y, HISTONE_MARKS


def generate_synthetic_data(n_samples=2000, n_bins=200):
    """Generate synthetic histone modification data."""
    np.random.seed(RANDOM_STATE)
    n_pos, n_neg = n_samples // 2, n_samples - n_samples // 2
    tss, x = n_bins // 2, np.arange(n_bins)
    X = []
    
    # H3K4me3 - TSS peak for active genes
    h3k4me3 = np.zeros((n_samples, n_bins))
    for i in range(n_pos):
        h3k4me3[i] = np.exp(-0.5 * ((x - tss) / 10) ** 2) * (3 + np.random.rand()) + np.random.randn(n_bins) * 0.2
    h3k4me3[n_pos:] = np.random.randn(n_neg, n_bins) * 0.3
    X.append(pd.DataFrame(h3k4me3, columns=[f'H3K4me3_{i}' for i in range(n_bins)]))
    
    # H3K4me1 - Enhancer mark
    X.append(pd.DataFrame(np.random.randn(n_samples, n_bins) * 0.5, 
                          columns=[f'H3K4me1_{i}' for i in range(n_bins)]))
    
    # H3K36me3 - Gene body for active genes
    h3k36me3 = np.zeros((n_samples, n_bins))
    for i in range(n_pos):
        h3k36me3[i, tss:] = 1.5 + np.random.randn(n_bins - tss) * 0.3
    h3k36me3[n_pos:] = np.random.randn(n_neg, n_bins) * 0.2
    X.append(pd.DataFrame(h3k36me3, columns=[f'H3K36me3_{i}' for i in range(n_bins)]))
    
    # H3K9me3, H3K27me3 - Repressive marks
    for mark in ['H3K9me3', 'H3K27me3']:
        data = np.random.randn(n_samples, n_bins) * 0.3
        data[n_pos:] += 0.8
        X.append(pd.DataFrame(data, columns=[f'{mark}_{i}' for i in range(n_bins)]))
    
    y = np.array([1] * n_pos + [0] * n_neg)
    idx = np.random.permutation(n_samples)
    X = [df.iloc[idx].reset_index(drop=True) for df in X]
    
    print(f"Generated: {n_samples} samples, {len(HISTONE_MARKS)} channels, {n_bins} bins")
    return X, y[idx], HISTONE_MARKS


def run_experiment():
    """Run REMC gene expression prediction experiment."""
    print("=" * 60)
    print("SublimeX: REMC Gene Expression Prediction")
    print("=" * 60)
    
    X, y, mark_names = load_remc_data()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    all_scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X[0], y), 1):
        print(f"\n--- Fold {fold}/{N_FOLDS} ---")
        
        X_train = [df.iloc[train_idx] for df in X]
        X_test = [df.iloc[test_idx] for df in X]
        y_train, y_test = y[train_idx], y[test_idx]
        
        sublimex = SublimeX(metric='auc', n_trials=N_TRIALS, verbose=True)
        features_train = sublimex.fit_transform(X_train, y_train)
        features_test = sublimex.transform(X_test)
        
        model = LightGBMModel(task='classification')
        score = model.test(features_train, y_train, features_test, y_test, 'auc')
        
        print(f"  Test AUC: {score:.4f}, Features: {len(sublimex.extracted_features)}")
        all_scores.append(score)
        
        if fold == 1:
            print("\n  Discovered features:")
            for desc in sublimex.get_feature_descriptions():
                print(f"    {desc}")
    
    print(f"\n{'='*60}")
    print(f"Mean AUC: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
    print("=" * 60)


if __name__ == '__main__':
    run_experiment()
