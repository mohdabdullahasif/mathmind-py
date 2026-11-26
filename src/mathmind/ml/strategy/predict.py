"""Strategy predictor – load a saved model and predict."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from mathmind.ml.strategy.features import extract_features

_MODEL_PATH = Path(__file__).parent.parent.parent.parent.parent / "models" / "strategy_model.pkl"
_model: Any = None


def _load_model() -> Any:
    global _model
    if _model is None:
        if _MODEL_PATH.exists():
            with open(_MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
        else:
            # fallback: rule-based prediction
            _model = None
    return _model


def _rule_based(features: np.ndarray) -> tuple[str, float]:
    """Simple rule-based fallback when model not trained."""
    is_eq, is_matrix, _, _, has_diff, has_integrate = features[:6]
    is_linear, is_quadratic = features[10:12]

    if is_matrix:
        return "matrix-inverse", 0.85
    if has_diff:
        return "differentiate", 0.85
    if has_integrate:
        return "integrate", 0.85
    if is_quadratic:
        return "quadratic-formula", 0.80
    if is_linear and is_eq:
        return "linear-isolate", 0.85
    return "direct-arithmetic", 0.75


def predict(problem: str) -> tuple[str, float, list[tuple[str, float]]]:
    """Predict strategy for *problem*.

    Returns (strategy, confidence, top2_alternatives).
    """
    features = extract_features(problem).reshape(1, -1)
    model = _load_model()

    if model is None:
        strategy, conf = _rule_based(features[0])
        return strategy, conf, []

    try:
        proba = model.predict_proba(features)[0]
        classes = model.classes_
        idx_sorted = np.argsort(proba)[::-1]
        top = [(classes[i], float(proba[i])) for i in idx_sorted[:3]]
        strategy, confidence = top[0]
        alternatives = top[1:]
        return strategy, confidence, alternatives
    except (ValueError, TypeError, AttributeError):
        strategy, conf = _rule_based(features[0])
        return strategy, conf, []
