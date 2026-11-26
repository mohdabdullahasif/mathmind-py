"""CNN model definition for math symbol recognition.

Architecture: simple CNN trained on 28x28 grayscale images.
Classes: 0-9, +, -, ×, ÷, =, (, ), x, y, z, ., /
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Class labels (index -> character)
CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "-", "*", "/", "=", "(", ")",
    "x", "y", "z", ".",
]
NUM_CLASSES = len(CLASSES)


def build_model() -> Any:
    """Build a PyTorch CNN model."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch not installed. Run: pip install torch torchvision")

    class SymbolCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, NUM_CLASSES),
            )

        def forward(self, x: Any) -> Any:
            return self.classifier(self.features(x))

    return SymbolCNN()


def load_model(model_path: str) -> Any:
    """Load a saved model from *model_path* (.pt file)."""
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch not installed")

    model = build_model()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
