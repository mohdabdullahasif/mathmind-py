"""__init__ for core package."""
from mathmind.core.engine import evaluate, solve
from mathmind.core.errors import (
    DivisionByZeroError,
    DomainError,
    EvalError,
    LexError,
    MathMindError,
    MatrixError,
    OverflowMathError,
    ParseError,
    UndefinedVariableError,
)
from mathmind.core.parser import parse
from mathmind.core.symbols import SymbolTable

__all__ = [
    "DivisionByZeroError",
    "DomainError",
    "EvalError",
    "LexError",
    "MathMindError",
    "MatrixError",
    "OverflowMathError",
    "ParseError",
    "SymbolTable",
    "UndefinedVariableError",
    "evaluate",
    "parse",
    "solve",
]
