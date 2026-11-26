"""Vision: image preprocessing and segmentation for handwritten math."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def _check_file(path: Path) -> None:
    from mathmind.core.errors import FileSizeError
    size = path.stat().st_size
    if size > MAX_FILE_SIZE:
        raise FileSizeError(size, MAX_FILE_SIZE)


def preprocess(image_path: str) -> Any:
    """Load, grayscale, denoise, threshold, deskew an image.
    Returns a binary numpy array (H x W) or raises ImportError if cv2 missing.
    """
    try:
        import cv2  # type: ignore[import]
    except ImportError:
        raise ImportError("opencv-python not installed. Run: pip install opencv-python")

    p = Path(image_path)
    _check_file(p)
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # denoise
    denoised = cv2.fastNlMeansDenoising(img, h=10)
    # adaptive threshold
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    # deskew
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        (h, w) = binary.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return binary


def segment_symbols(binary: Any) -> list[Any]:
    """Return list of 28x28 crops (sorted left-to-right) for each symbol."""
    try:
        import cv2  # type: ignore[import]
    except ImportError:
        raise ImportError("opencv-python not installed")

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 20:
            continue  # skip tiny noise
        boxes.append((x, y, w, h))

    # sort left-to-right
    boxes.sort(key=lambda b: b[0])

    crops = []
    for (x, y, w, h) in boxes:
        crop = binary[y:y + h, x:x + w]
        # pad to square
        size = max(w, h)
        padded = np.zeros((size, size), dtype=np.uint8)
        offset_x = (size - w) // 2
        offset_y = (size - h) // 2
        padded[offset_y:offset_y + h, offset_x:offset_x + w] = crop
        # resize to 28x28
        resized = cv2.resize(padded, (28, 28))
        crops.append(resized)

    return crops
