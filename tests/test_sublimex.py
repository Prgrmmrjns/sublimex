#!/usr/bin/env python3
"""Tests for SublimeX package using REMC data."""

import numpy as np
import pandas as pd
import sys
import os
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'processed_datasets'
)
IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')


def load_remc(cell_line='E003'):
    """Load REMC dataset."""
    df = pd.read_parquet(f"{DATASET_FOLDER}/remc/{cell_line}.parquet")
    histone_marks = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
    X = [df[[c for c in df.columns if c.startswith(f"{s}_")]].copy() for s in histone_marks]
    return X, df['target']


def test_full_pipeline():
    """Test complete SublimeX pipeline on REMC data."""
    print("\n=== Test: Full Pipeline ===")
    
    from sublimex import SublimeX
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb
    
    X, y = load_remc()
    print(f"  Data: {len(y)} samples, {len(X)} channels, {X[0].shape[1]} bins")
    
    # Split
    idx_train, idx_test = train_test_split(range(len(y)), test_size=0.2, stratify=y, random_state=42)
    X_train = [x.iloc[idx_train].astype(np.float32) for x in X]
    X_test = [x.iloc[idx_test].astype(np.float32) for x in X]
    y_train, y_test = y.iloc[idx_train].values, y.iloc[idx_test].values
    
    # Fit
    model = SublimeX(metric='auc', n_trials=50, verbose=True)
    train_features = model.fit_transform(X_train, y_train)
    test_features = model.transform(X_test)
    
    print(f"  Features: {train_features.shape[1]}")
    
    # Evaluate
    clf = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
    clf.fit(train_features, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(test_features)[:, 1])
    print(f"  Test AUC: {auc:.4f}")
    
    assert 0.5 < auc < 1.0
    print("  PASSED!")
    return model, X_train, y_train, train_features, clf


def test_save_load():
    """Test save/load features."""
    print("\n=== Test: Save/Load ===")
    
    from sublimex import SublimeX
    
    X, y = load_remc()
    X_small = [x.iloc[:200].astype(np.float32) for x in X]
    y_small = y.iloc[:200].values
    
    model = SublimeX(metric='auc', n_trials=30)
    model.fit(X_small, y_small)
    
    path = '/tmp/test_features.json'
    model.save_features(path)
    
    new_model = SublimeX()
    new_model.load_features(path)
    features = new_model.transform(X_small)
    
    assert new_model._is_fitted
    assert features.shape[1] == len(model.extracted_features)
    os.remove(path)
    print("  PASSED!")


def test_custom_components():
    """Test custom transforms, objectives, and models."""
    print("\n=== Test: Custom Components ===")
    
    from sublimex import SublimeX, register_transform, create_custom_objective, SklearnModelWrapper
    from sklearn.linear_model import LogisticRegression
    
    X, y = load_remc()
    X_small = [x.iloc[:200].astype(np.float32) for x in X]
    y_small = y.iloc[:200].values
    
    # Custom transform
    register_transform('abs', lambda x: np.abs(x), overwrite=True)
    
    # Custom objective
    rms_obj = create_custom_objective(lambda s: np.sqrt((s**2).mean(axis=1, keepdims=True)), 'rms')
    
    # Custom model
    model = SublimeX(
        metric='auc', n_trials=30,
        transforms={'raw': lambda x: x, 'abs': np.abs},
        objective_fn=rms_obj,
        model=SklearnModelWrapper(LogisticRegression(max_iter=1000))
    )
    model.fit(X_small, y_small)
    
    assert model._is_fitted
    print("  PASSED!")


def test_visualizations(model, X_train, y_train, train_features, clf):
    """Test visualization functions."""
    print("\n=== Test: Visualizations ===")
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sublimex.visualization import (
        plot_feature_importance, plot_segment_on_signal,
        plot_feature_distributions, plot_transform_comparison, plot_optimization_history
    )
    from sublimex import TRANSFORMS
    
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    
    descriptions = model.get_feature_descriptions()
    signal = X_train[0].values if hasattr(X_train[0], 'values') else X_train[0]
    
    # Feature importance
    fig = plot_feature_importance(clf.feature_importances_, descriptions)
    fig.savefig(f'{IMAGES_FOLDER}/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Segment on signal
    if model.extracted_features:
        fig = plot_segment_on_signal(signal, model.extracted_features[0], signal.shape[1], model.transform_names)
        fig.savefig(f'{IMAGES_FOLDER}/segment_on_signal.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Feature distributions
    fig = plot_feature_distributions(train_features, y_train, 0, descriptions[0] if descriptions else None)
    fig.savefig(f'{IMAGES_FOLDER}/feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Transform comparison
    fig = plot_transform_comparison(signal, TRANSFORMS)
    fig.savefig(f'{IMAGES_FOLDER}/transform_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Optimization history
    fig = plot_optimization_history(0.5 + 0.3 * (1 - np.exp(-np.arange(50) / 15)))
    fig.savefig(f'{IMAGES_FOLDER}/optimization_history.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    saved = [f for f in os.listdir(IMAGES_FOLDER) if f.endswith('.png')]
    print(f"  Saved {len(saved)} images to {IMAGES_FOLDER}/")
    assert len(saved) >= 5
    print("  PASSED!")


if __name__ == '__main__':
    print("=" * 60)
    print("SublimeX Test Suite")
    print("=" * 60)
    
    model, X_train, y_train, train_features, clf = test_full_pipeline()
    test_save_load()
    test_custom_components()
    test_visualizations(model, X_train, y_train, train_features, clf)
    
    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
