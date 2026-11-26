"""Tests for the core expression engine (MVP 1-2)."""
from __future__ import annotations

import math
import pytest

from mathmind.core.parser import parse
from mathmind.core.evaluator import Evaluator
from mathmind.core.symbols import SymbolTable
from mathmind.core.engine import solve, evaluate
from mathmind.core.errors import (
    DivisionByZeroError, LexError, ParseError, UndefinedVariableError, DomainError
)


def _eval(expr: str, sym: SymbolTable | None = None) -> float:
    return evaluate(expr, sym or SymbolTable())


# ---- arithmetic precedence ----
@pytest.mark.parametrize("expr,expected", [
    ("2 + 3", 5.0),
    ("10 - 4", 6.0),
    ("3 * 4", 12.0),
    ("10 / 4", 2.5),
    ("10 % 3", 1.0),
    ("2 ** 10", 1024.0),
    ("2 ^ 8", 256.0),
    ("2 + 3 * 4", 14.0),
    ("(2 + 3) * 4", 20.0),
    ("-2 ** 2", -4.0),   # unary minus: -(2**2)
    ("2 ** 2 ** 3", 256.0),  # right-assoc: 2**(2**3)
    ("1 + 2 + 3 + 4", 10.0),
    ("100 / 10 / 2", 5.0),
])
def test_arithmetic(expr: str, expected: float) -> None:
    assert _eval(expr) == pytest.approx(expected)


# ---- unary minus ----
def test_unary_minus() -> None:
    assert _eval("-5") == -5.0
    assert _eval("--5") == 5.0
    assert _eval("-(3+2)") == -5.0


# ---- implicit multiplication ----
@pytest.mark.parametrize("expr,expected", [
    ("2(3+1)", 8.0),
    ("3(2)", 6.0),
])
def test_implicit_mul(expr: str, expected: float) -> None:
    assert _eval(expr) == pytest.approx(expected)


# ---- variables ----
def test_variables() -> None:
    sym = SymbolTable()
    evaluate("x = 5", sym)
    assert _eval("x * 2", sym) == 10.0


def test_ans() -> None:
    sym = SymbolTable()
    evaluate("3 + 7", sym)
    assert sym.ans == 10.0
    assert _eval("ans + 1", sym) == 11.0


# ---- built-in functions ----
@pytest.mark.parametrize("expr,expected", [
    ("sqrt(16)", 4.0),
    ("abs(-7)", 7.0),
    ("factorial(5)", 120.0),
    ("floor(3.7)", 3.0),
    ("ceil(3.2)", 4.0),
    ("round(3.5)", 4.0),
    ("min(1,2,3)", 1.0),
    ("max(1,2,3)", 3.0),
    ("log(1)", 0.0),
    ("ln(1)", 0.0),
])
def test_builtins(expr: str, expected: float) -> None:
    assert _eval(expr) == pytest.approx(expected, rel=1e-6)


# ---- trig (radians) ----
def test_trig_radians() -> None:
    sym = SymbolTable(angle_mode="radians")
    assert _eval("sin(0)", sym) == pytest.approx(0.0)
    assert _eval("cos(0)", sym) == pytest.approx(1.0)
    assert _eval("sin(pi)", sym) == pytest.approx(0.0, abs=1e-10)


# ---- trig (degrees) ----
def test_trig_degrees() -> None:
    sym = SymbolTable(angle_mode="degrees")
    assert _eval("sin(90)", sym) == pytest.approx(1.0)
    assert _eval("cos(0)", sym) == pytest.approx(1.0)
    assert _eval("tan(45)", sym) == pytest.approx(1.0)


# ---- constants ----
def test_constants() -> None:
    assert _eval("pi") == pytest.approx(math.pi)
    assert _eval("e") == pytest.approx(math.e)


# ---- user functions ----
def test_user_function() -> None:
    sym = SymbolTable()
    evaluate("f(x) = x**2 + 1", sym)
    assert _eval("f(3)", sym) == pytest.approx(10.0)
    assert _eval("f(3) + 2", sym) == pytest.approx(12.0)


def test_nested_user_function() -> None:
    sym = SymbolTable()
    evaluate("f(x) = x * 2", sym)
    evaluate("g(x) = f(x) + 1", sym)
    assert _eval("g(5)", sym) == pytest.approx(11.0)


# ---- errors ----
def test_division_by_zero() -> None:
    with pytest.raises(Exception):
        _eval("1 / 0")


def test_undefined_variable() -> None:
    with pytest.raises(Exception):
        _eval("undefined_var_xyz")


def test_lex_error() -> None:
    with pytest.raises(Exception):
        _eval("2 @ 3")


def test_parse_error() -> None:
    with pytest.raises(Exception):
        _eval("(2 + 3")


def test_sqrt_negative() -> None:
    r = solve("sqrt(-1)")
    assert r.error is not None


# ---- matrix ----
def test_matrix_literal() -> None:
    import numpy as np
    sym = SymbolTable()
    result = evaluate("[[1,2],[3,4]]", sym)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)


def test_matrix_add() -> None:
    import numpy as np
    sym = SymbolTable()
    a = evaluate("[[1,0],[0,1]]", sym)
    b = evaluate("[[2,3],[4,5]]", sym)
    # direct eval
    r = solve("[[1,0],[0,1]] + [[2,3],[4,5]]", sym)
    assert r.error is None


# ---- result object ----
def test_result_has_steps() -> None:
    r = solve("2 + 2")
    assert r.value == 4.0
    assert len(r.steps) > 0
    assert r.error is None


def test_result_has_latex() -> None:
    r = solve("2 + 3")
    assert r.latex != ""


# ---- simplifier ----
def test_simplifier_x_plus_zero() -> None:
    from mathmind.core.parser import parse
    from mathmind.core.simplifier import simplify
    from mathmind.core.ast import NumberNode, IdentNode, BinaryNode
    node = BinaryNode("+", IdentNode("x"), NumberNode(0.0))
    simplified = simplify(node)
    assert isinstance(simplified, IdentNode)
    assert simplified.name == "x"


def test_simplifier_x_times_one() -> None:
    from mathmind.core.parser import parse
    from mathmind.core.simplifier import simplify
    from mathmind.core.ast import NumberNode, IdentNode, BinaryNode
    node = BinaryNode("*", IdentNode("x"), NumberNode(1.0))
    simplified = simplify(node)
    assert isinstance(simplified, IdentNode)


def test_simplifier_x_minus_x() -> None:
    from mathmind.core.parser import parse
    from mathmind.core.simplifier import simplify
    from mathmind.core.ast import NumberNode, IdentNode, BinaryNode
    node = BinaryNode("-", IdentNode("x"), IdentNode("x"))
    simplified = simplify(node)
    assert isinstance(simplified, NumberNode)
    assert simplified.value == 0.0


def test_simplifier_constant_folding() -> None:
    from mathmind.core.simplifier import simplify
    from mathmind.core.parser import parse
    node = parse("2 + 3 * 4")
    simplified = simplify(node)
    from mathmind.core.ast import NumberNode
    assert isinstance(simplified, NumberNode)
    assert simplified.value == 14.0


# ---- fuzz ----
@pytest.mark.parametrize("expr", [
    "", ")", "(", "***", "1 + + +", "++1", "1//2", "1@2", "x *** y",
    "sin()", "f(1,2", "[[1,2]", "1 2 3", "...", "1e",
])
def test_fuzz_no_crash(expr: str) -> None:
    """Parser must not crash on garbage input – just raise an exception."""
    try:
        parse(expr)
    except Exception:
        pass  # expected
