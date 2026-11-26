"""Train a lightweight scikit-learn symbol classifier for 28x28 crops.

Used as a zero-PyTorch fallback (e.g. on Streamlit Cloud) when
`models/symbol_cnn.pt` is not present.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

from mathmind.inputs.vision.model import CLASSES

_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
]

_DEFAULT_OUT = Path(__file__).parent / "symbol_clf.joblib"


def _crop_and_resize(canvas: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(canvas > 0))
    if len(coords) == 0:
        return np.zeros((28, 28), dtype=np.uint8)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    crop = canvas[y0:y1, x0:x1]
    h, w = crop.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=np.uint8)
    ox = (size - w) // 2
    oy = (size - h) // 2
    padded[oy : oy + h, ox : ox + w] = crop
    return cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)


def _draw_hand_variant(sym: str, thickness: int) -> np.ndarray | None:
    """Draw extra hand-drawn style variants for common math symbols."""
    canvas = np.zeros((64, 64), dtype=np.uint8)
    if sym == "1":
        dx = random.randint(-3, 3)
        cv2.line(canvas, (32 + dx, 12), (32 - dx, 52), 255, thickness)
        if random.random() < 0.4:
            cv2.line(canvas, (24, 20), (32 + dx, 12), 255, thickness)
        return canvas
    if sym == "+":
        cx, cy = 32 + random.randint(-2, 2), 32 + random.randint(-2, 2)
        arm = random.randint(14, 20)
        cv2.line(canvas, (cx - arm, cy), (cx + arm, cy), 255, thickness)
        cv2.line(canvas, (cx, cy - arm), (cx, cy + arm), 255, thickness)
        return canvas
    if sym == "-":
        y = 32 + random.randint(-3, 3)
        cv2.line(canvas, (14, y), (50, y), 255, thickness)
        return canvas
    if sym == "/":
        cv2.line(canvas, (44, 12), (20, 52), 255, thickness)
        return canvas
    if sym == ".":
        cv2.circle(canvas, (32, 32), random.randint(3, 5), 255, -1)
        return canvas
    return None


def _render_sample(sym: str) -> np.ndarray:
    thickness = random.randint(1, 3)
    if random.random() < 0.35:
        hand = _draw_hand_variant(sym, thickness)
        if hand is not None:
            canvas = hand
        else:
            canvas = np.zeros((64, 64), dtype=np.uint8)
            font = random.choice(_FONTS)
            scale = random.uniform(1.0, 1.5)
            cv2.putText(canvas, sym, (14, 48), font, scale, 255, thickness, cv2.LINE_AA)
    else:
        canvas = np.zeros((64, 64), dtype=np.uint8)
        font = random.choice(_FONTS)
        scale = random.uniform(1.0, 1.5)
        cv2.putText(canvas, sym, (14, 48), font, scale, 255, thickness, cv2.LINE_AA)

    angle = random.uniform(-14, 14)
    M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
    rotated = cv2.warpAffine(canvas, M, (64, 64))
    return _crop_and_resize(rotated)


def extract_crop_features(crop: np.ndarray) -> np.ndarray:
    """Convert a 28x28 grayscale crop to a normalized feature vector."""
    norm = crop.astype(np.float32) / 255.0
    if norm.max() > 0:
        norm = norm / norm.max()
    flat = norm.flatten()
    h_proj = norm.mean(axis=1)
    v_proj = norm.mean(axis=0)
    return np.concatenate([flat, h_proj, v_proj])


def train_fallback_model(
    samples_per_class: int = 180, out_path: Path | None = None
) -> Path:
    random.seed(42)
    np.random.seed(42)

    X_list = []
    y_list = []
    for idx, sym in enumerate(CLASSES):
        for _ in range(samples_per_class):
            crop = _render_sample(sym)
            X_list.append(extract_crop_features(crop))
            y_list.append(idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    clf = MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=60,
        random_state=42,
        early_stopping=True,
    )
    clf.fit(X, y)

    dest = out_path or _DEFAULT_OUT
    joblib.dump(clf, dest, compress=3)
    return dest


if __name__ == "__main__":
    p = train_fallback_model()
    print(f"Saved fallback symbol classifier to {p} ({p.stat().st_size / 1024:.1f} KB)")
