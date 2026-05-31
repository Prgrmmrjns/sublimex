"""SublimeX: Supervised Bottom-Up Localized Multi-Representative Feature Extraction."""

__version__ = "0.1.2"
__author__ = "J.C. Wolber"

from sublimex.core import (
    SublimeX, FeatureExtractor, mean_objective, random_split_mean_objective, aggregate_objective, pattern_objective,
    decision_tree_objective, TRANSFORMS, extract_feature, extract_feature_from_params,
    suggest_segment_params, AGGREGATIONS,
    _get_segment, _evaluate, _suggest_segment, _get_train_val_split,
)
from sublimex.transforms import TRANSFORMS as PKG_TRANSFORMS
from sublimex.objectives import mean_objective as pkg_mean_objective
from sublimex.models import LightGBMModel, encode_labels
from sublimex.visualization import plot_segment_on_signal

__all__ = [
    "__version__", "__author__", "FeatureExtractor", "SublimeX", "TRANSFORMS", "PKG_TRANSFORMS",
    "mean_objective", "random_split_mean_objective", "aggregate_objective", "pattern_objective", "decision_tree_objective",
    "pkg_mean_objective", "extract_feature", "extract_feature_from_params", "suggest_segment_params",
    "AGGREGATIONS", "LightGBMModel", "encode_labels",
    "_get_segment", "_evaluate", "_suggest_segment", "_get_train_val_split",
    "plot_segment_on_signal",
]
