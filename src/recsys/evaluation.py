"""Evaluation convenience functions."""
from __future__ import annotations

from lightfm import LightFM

from . import config, data, model as model_module


def evaluate(
    prepared: data.PreparedData,
    trained_model: LightFM,
    k: int = config.DEFAULT_TOP_K,
) -> model_module.EvaluationResults:
    """Wrapper around :func:`model.evaluate_model` for convenience."""
    return model_module.evaluate_model(trained_model, prepared, k=k)
