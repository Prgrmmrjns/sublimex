"""SublimeX: Supervised Bottom-Up Localized Multi-Representative Feature Extraction."""

__version__ = "0.1.1"
__author__ = "J.C. Wolber"

from sublimex.core import SublimeX
from sublimex.transforms import TRANSFORMS
from sublimex.objectives import mean_objective
from sublimex.models import LightGBMModel
from sublimex.visualization import plot_segment_on_signal

__all__ = [
    "__version__", "__author__", "SublimeX", "TRANSFORMS",
    "mean_objective", "LightGBMModel",
    "plot_segment_on_signal",
]
