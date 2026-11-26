"""Shared data models for MathMind."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    """One step in a solution."""
    step_no: int
    description: str
    before: str
    after: str


@dataclass
class Result:
    """Return value of solve()."""
    value: Any
    steps: list[Step] = field(default_factory=list)
    latex: str = ""
    strategy: str = ""
    confidence: float = 1.0
    expression: str = ""
    channel: str = "typed"
    error: str | None = None
