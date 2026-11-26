"""High-level engine combining parse + simplify + evaluate.

This is the main entry point: solve(expression) -> Result.
"""
from __future__ import annotations

from typing import Any

from mathmind.core.errors import MathMindError
from mathmind.core.evaluator import Evaluator
from mathmind.core.parser import parse
from mathmind.core.simplifier import simplify
from mathmind.core.symbols import SymbolTable
from mathmind.models import Result, Step


_DEFAULT_SYMBOLS = SymbolTable()


def solve(
    expression: str,
    symbols: SymbolTable | None = None,
    channel: str = "typed",
    strategy: str = "",
    confidence: float = 1.0,
) -> Result:
    """Parse, simplify, evaluate *expression* and return a Result."""
    if symbols is None:
        symbols = _DEFAULT_SYMBOLS

    steps: list[Step] = []
    try:
        ast = parse(expression)
        steps.append(Step(1, "Parse expression", expression, "AST built"))

        simplified = simplify(ast)
        steps.append(Step(2, "Simplify AST", "raw AST", "simplified AST"))

        evaluator = Evaluator(symbols)
        value = evaluator.eval(simplified)

        # update ans
        if isinstance(value, (int, float)):
            symbols.set_var("ans", float(value))

        from mathmind.export.latex import ast_to_latex
        latex = ast_to_latex(simplified)

        return Result(
            value=value,
            steps=steps,
            latex=latex,
            strategy=strategy,
            confidence=confidence,
            expression=expression,
            channel=channel,
        )
    except MathMindError as exc:
        return Result(
            value=None,
            steps=steps,
            expression=expression,
            channel=channel,
            error=str(exc),
        )
    except Exception as exc:
        return Result(
            value=None,
            steps=steps,
            expression=expression,
            channel=channel,
            error=f"Unexpected error: {exc}",
        )


def evaluate(expression: str, symbols: SymbolTable | None = None) -> Any:
    """Quick evaluate – returns raw value or raises on error."""
    r = solve(expression, symbols)
    if r.error:
        raise MathMindError(r.error)
    return r.value
