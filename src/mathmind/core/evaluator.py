"""AST evaluator for MathMind.

Walks the AST produced by parser.py and returns a numeric (or matrix) result.
No eval() or exec() used anywhere.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from mathmind.core.ast import (
    ASTNode,
    AssignNode,
    BinaryNode,
    CallNode,
    IdentNode,
    MatrixNode,
    NumberNode,
    UnaryNode,
)
from mathmind.core.errors import (
    DivisionByZeroError,
    DomainError,
    EvalError,
    OverflowMathError,
    UndefinedVariableError,
)
from mathmind.core.symbols import BUILTINS, SymbolTable

_TRIG_NAMES = {"sin", "cos", "tan", "asin", "acos", "atan"}


class Evaluator:
    def __init__(self, symbols: SymbolTable) -> None:
        self._sym = symbols

    def eval(self, node: ASTNode) -> Any:
        match node:
            case NumberNode(value=v):
                return v
            case IdentNode(name=n, col=c):
                return self._sym.get_var(n)
            case UnaryNode(op=op, operand=operand):
                v = self.eval(operand)
                return -v if op == "-" else v
            case BinaryNode(op=op, left=left, right=right):
                return self._eval_binary(op, left, right)
            case CallNode(name=name, args=args, col=col):
                return self._eval_call(name, args, col)
            case AssignNode(name=name, params=params, body=body):
                return self._eval_assign(name, params, body)
            case MatrixNode(rows=rows):
                return self._eval_matrix(rows)
            case _:
                raise EvalError(f"Unknown AST node: {type(node)}")

    def _eval_binary(self, op: str, left: ASTNode, right: ASTNode) -> Any:
        lv = self.eval(left)
        rv = self.eval(right)

        # matrix ops
        if isinstance(lv, np.ndarray) or isinstance(rv, np.ndarray):
            return self._matrix_op(op, lv, rv)

        try:
            match op:
                case "+":
                    return lv + rv
                case "-":
                    return lv - rv
                case "*":
                    return lv * rv
                case "/":
                    if rv == 0:
                        raise DivisionByZeroError()
                    return lv / rv
                case "%":
                    if rv == 0:
                        raise DivisionByZeroError()
                    return lv % rv
                case "**":
                    if lv == 0 and rv < 0:
                        raise DivisionByZeroError()
                    if lv < 0 and not float(rv).is_integer():
                        raise DomainError("Negative base with fractional exponent")
                    result = lv ** rv
                    if math.isinf(result):
                        raise OverflowMathError()
                    return result
                case _:
                    raise EvalError(f"Unknown operator '{op}'")
        except (OverflowError, ValueError) as exc:
            raise EvalError(str(exc)) from exc

    def _matrix_op(self, op: str, lv: Any, rv: Any) -> np.ndarray:
        match op:
            case "+":
                return np.add(lv, rv)
            case "-":
                return np.subtract(lv, rv)
            case "*":
                if isinstance(lv, np.ndarray) and isinstance(rv, np.ndarray):
                    return np.matmul(lv, rv)
                return lv * rv
            case _:
                raise EvalError(f"Operator '{op}' not supported on matrices")

    def _eval_call(self, name: str, args: list[ASTNode], col: int) -> Any:
        evaluated = [self.eval(a) for a in args]
        fn_or_def = self._sym.get_func(name)

        # built-in callable
        if callable(fn_or_def):
            fn = fn_or_def
            try:
                # degree-mode trig
                if name in _TRIG_NAMES and len(evaluated) == 1:
                    return self._sym.call_trig(fn, evaluated[0])
                return fn(*evaluated)
            except (ValueError, TypeError, OverflowError) as exc:
                raise DomainError(str(exc)) from exc

        # user-defined function
        params, body = fn_or_def  # type: ignore[misc]
        if len(params) != len(evaluated):
            raise EvalError(
                f"Function '{name}' takes {len(params)} args, got {len(evaluated)}"
            )
        # push a child scope
        child = SymbolTable(angle_mode=self._sym.angle_mode)
        child._vars = dict(self._sym._vars)
        child._funcs = dict(self._sym._funcs)
        for p, v in zip(params, evaluated):
            child.set_var(p, v)
        return Evaluator(child).eval(body)

    def _eval_assign(self, name: str, params: list[str], body: ASTNode) -> Any:
        if params:
            self._sym.set_func(name, params, body)
            return f"Function '{name}' defined"
        else:
            value = self.eval(body)
            self._sym.set_var(name, value)
            return value

    def _eval_matrix(self, rows: list[list[ASTNode]]) -> np.ndarray:
        return np.array([[self.eval(cell) for cell in row] for row in rows], dtype=float)
