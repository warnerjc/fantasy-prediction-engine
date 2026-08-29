"""Training + inference. Consumes /features rows and /scoring-computed labels.

Config-driven (target column, objective, temporal split are all ``ModelConfig``
inputs, never hardcoded). Walk-forward validation only. One model per position.
Predictions are always ``Prediction(mean, p10, p50, p90)`` — v1 fills ``mean``
and leaves the quantiles ``None``.
"""

from .config import ModelConfig, DEFAULT_CONFIGS
from .labels import season_labels
from .prediction import Prediction
from .pipeline import assemble_position, project_position, walk_forward

__all__ = [
    "ModelConfig",
    "DEFAULT_CONFIGS",
    "season_labels",
    "Prediction",
    "assemble_position",
    "project_position",
    "walk_forward",
]
