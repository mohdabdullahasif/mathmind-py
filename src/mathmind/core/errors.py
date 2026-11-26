"""Custom exceptions for MathMind."""
from __future__ import annotations


class MathMindError(Exception):
    """Base error."""


class LexError(MathMindError):
    def __init__(self, message: str, col: int) -> None:
        super().__init__(f"LexError at col {col}: {message}")
        self.col = col


class ParseError(MathMindError):
    def __init__(self, message: str, col: int = 0) -> None:
        super().__init__(f"ParseError at col {col}: {message}")
        self.col = col


class EvalError(MathMindError):
    """Runtime evaluation error."""


class DivisionByZeroError(EvalError):
    def __init__(self) -> None:
        super().__init__("Division by zero")


class DomainError(EvalError):
    def __init__(self, message: str) -> None:
        super().__init__(f"Domain error: {message}")


class OverflowMathError(EvalError):
    def __init__(self) -> None:
        super().__init__("Numeric overflow")


class UndefinedVariableError(EvalError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Undefined variable or function: '{name}'")


class MatrixError(MathMindError):
    """Matrix operation error."""


class FileSizeError(MathMindError):
    """Uploaded file too large."""
    def __init__(self, size: int, max_size: int) -> None:
        super().__init__(f"File size {size} bytes exceeds limit {max_size} bytes")
