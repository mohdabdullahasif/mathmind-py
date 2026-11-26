"""Symbol table for MathMind REPL.

Stores variables, user-defined functions, and built-ins.
No eval() or exec() used anywhere.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from mathmind.core.errors import UndefinedVariableError


# ------------------------------------------------------------------ built-ins
def _safe_sqrt(x: float) -> float:
    if x < 0:
        raise ValueError("sqrt of negative number")
    return math.sqrt(x)


def _safe_log(x: float, base: float = math.e) -> float:
    if x <= 0:
        raise ValueError("log of non-positive number")
    return math.log(x, base)


def _safe_ln(x: float) -> float:
    if x <= 0:
        raise ValueError("ln of non-positive number")
    return math.log(x)


def _safe_factorial(x: float) -> int:
    if x < 0 or not float(x).is_integer():
        raise ValueError("factorial requires a non-negative integer")
    return math.factorial(int(x))


BUILTINS: dict[str, Any] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "log": _safe_log,
    "ln": _safe_ln,
    "sqrt": _safe_sqrt,
    "abs": abs,
    "factorial": _safe_factorial,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "min": min,
    "max": max,
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "ans": 0.0,
}


class SymbolTable:
    """Hierarchical scope: builtins → global → local."""

    def __init__(self, angle_mode: str = "radians") -> None:
        self._vars: dict[str, float] = {}
        self._funcs: dict[str, tuple[list[str], Any]] = {}
        self.angle_mode = angle_mode  # 'radians' or 'degrees'
        self.ans: float = 0.0

    # ---------------------------------------------------------------- variables
    def set_var(self, name: str, value: float) -> None:
        self._vars[name] = value
        if name == "ans":
            self.ans = value

    def get_var(self, name: str) -> float:
        if name == "ans":
            return self.ans
        if name in self._vars:
            return self._vars[name]
        if name in BUILTINS and not callable(BUILTINS[name]):
            return float(BUILTINS[name])
        raise UndefinedVariableError(name)

    def has_var(self, name: str) -> bool:
        return name in self._vars or (
            name in BUILTINS and not callable(BUILTINS[name])
        )

    # ---------------------------------------------------------------- functions
    def set_func(self, name: str, params: list[str], body: Any) -> None:
        self._funcs[name] = (params, body)

    def get_func(self, name: str) -> tuple[list[str], Any] | Callable:  # type: ignore[return]
        if name in self._funcs:
            return self._funcs[name]
        if name in BUILTINS and callable(BUILTINS[name]):
            return BUILTINS[name]  # type: ignore[return-value]
        raise UndefinedVariableError(name)

    def has_func(self, name: str) -> bool:
        return name in self._funcs or (name in BUILTINS and callable(BUILTINS[name]))

    def _trig_input(self, x: float) -> float:
        if self.angle_mode == "degrees":
            return math.radians(x)
        return x

    def call_trig(self, fn: Callable, x: float) -> float:  # type: ignore[type-arg]
        return fn(self._trig_input(x))

    def all_names(self) -> list[str]:
        return list(BUILTINS.keys()) + list(self._vars.keys()) + list(self._funcs.keys())
