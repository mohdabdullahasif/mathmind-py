"""Symbol inference: run CNN on image crops and return predictions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mathmind.inputs.vision.model import CLASSES

_FALLBACK_MODEL_PATH = Path(__file__).parent / "symbol_clf.joblib"
_fallback_clf: Any = None


def _get_fallback_clf() -> Any:
    global _fallback_clf
    if _fallback_clf is None:
        import joblib

        if not _FALLBACK_MODEL_PATH.exists():
            from mathmind.inputs.vision.train_fallback import train_fallback_model

            train_fallback_model(out_path=_FALLBACK_MODEL_PATH)
        _fallback_clf = joblib.load(_FALLBACK_MODEL_PATH)
    return _fallback_clf


def predict_symbols_fallback(
    crops: list[Any],
    top_k: int = 3,
) -> list[list[tuple[str, float]]]:
    """Predict top-k (label, confidence) per crop using the lightweight sklearn model."""
    if not crops:
        return []
    from mathmind.inputs.vision.train_fallback import extract_crop_features

    clf = _get_fallback_clf()
    X = np.array([extract_crop_features(c) for c in crops], dtype=np.float32)
    probs = clf.predict_proba(X)
    results = []
    for prob_row in probs:
        indices = np.argsort(prob_row)[::-1][:top_k]
        results.append([(CLASSES[i], float(prob_row[i])) for i in indices])
    return results


def predict_symbols(
    crops: list[Any],
    model: Any,
    top_k: int = 3,
) -> list[list[tuple[str, float]]]:
    """Run *model* on each crop and return top-k (label, confidence) per symbol."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("PyTorch not installed")

    if not crops:
        return []

    # stack crops into a batch tensor (N, 1, 28, 28)
    batch = np.stack([c.astype(np.float32) / 255.0 for c in crops])
    tensor = torch.from_numpy(batch).unsqueeze(1)  # (N, 1, 28, 28)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1).numpy()

    results = []
    for prob_row in probs:
        indices = np.argsort(prob_row)[::-1][:top_k]
        preds = [(CLASSES[i], float(prob_row[i])) for i in indices]
        results.append(preds)

    return results


def predictions_to_expression(
    predictions: list[list[tuple[str, float]]],
) -> tuple[str, float]:
    """Convert per-symbol top-1 predictions to an expression string.

    Returns (expression, joint_confidence).
    """
    if not predictions:
        return "", 0.0

    chars = []
    confidences = []
    for symbol_preds in predictions:
        label, conf = symbol_preds[0]
        chars.append(label)
        confidences.append(conf)

    expr = "".join(chars)
    joint_conf = float(np.prod(confidences)) if confidences else 0.0
    return expr, joint_conf


def beam_search_expression(
    predictions: list[list[tuple[str, float]]],
    parse_fn: Any,
    beam_width: int = 5,
) -> tuple[str, float]:
    """Beam search over top-k predictions to find a parseable expression.

    Falls back to greedy if no beam candidate parses.
    """
    if not predictions:
        return "", 0.0

    # initialize beams: list of (log_prob, [chars])
    beams: list[tuple[float, list[str]]] = [(0.0, [])]

    for symbol_preds in predictions:
        new_beams: list[tuple[float, list[str]]] = []
        for log_prob, chars in beams:
            for label, conf in symbol_preds[:beam_width]:
                import math
                new_log_prob = log_prob + math.log(max(conf, 1e-9))
                new_beams.append((new_log_prob, chars + [label]))
        # keep best beam_width
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

    from mathmind.core.errors import MathMindError

    # find first parseable candidate
    for log_prob, chars in beams:
        expr = "".join(chars)
        try:
            parse_fn(expr)
            import math
            conf = math.exp(log_prob / max(len(chars), 1))
            return expr, conf
        except MathMindError:
            continue

    # greedy fallback
    greedy_chars = [preds[0][0] for preds in predictions]
    return "".join(greedy_chars), float(np.prod([p[0][1] for p in predictions]))
