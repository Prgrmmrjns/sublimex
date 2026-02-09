"""SublimeX: Supervised Bottom-Up Localized Multi-Representative Feature Extraction."""

__version__ = "0.1.1"
__author__ = "J.C. Wolber"

from sublimex.core import SublimeX
from sublimex.transforms import TRANSFORMS, register_transform, get_transform, list_transforms
from sublimex.objectives import (
    AGGREGATIONS,
    mean_objective,
    aggregate_objective,
    pattern_objective,
    default_objective,
    create_custom_objective,
)
from sublimex.models import LightGBMModel, SklearnModelWrapper
from sublimex.visualization import (plot_feature_importance, plot_segment_on_signal,
                                    plot_feature_distributions, plot_transform_comparison,
                                    plot_optimization_history)

__all__ = [
    "__version__",
    "__author__",
    "SublimeX",
    "TRANSFORMS",
    "register_transform",
    "get_transform",
    "list_transforms",
    "AGGREGATIONS",
    "mean_objective",
    "aggregate_objective",
    "pattern_objective",
    "default_objective",
    "create_custom_objective",
    "LightGBMModel",
    "SklearnModelWrapper",
    "plot_feature_importance",
    "plot_segment_on_signal",
    "plot_feature_distributions",
    "plot_transform_comparison",
    "plot_optimization_history",
]
