"""Tests for symbolic math, matrix, stats, NL, strategy."""
from __future__ import annotations

import pytest


# ---- symbolic ----
def test_symbolic_solve_linear() -> None:
    from mathmind.math.symbolic import symbolic_solve
    r = symbolic_solve("2*x + 3 = 7")
    assert r.error is None
    assert "x = 2" in r.value or "2" in str(r.value)


def test_symbolic_solve_quadratic() -> None:
    from mathmind.math.symbolic import symbolic_solve
    r = symbolic_solve("x**2 - 5*x + 6 = 0")
    assert r.error is None
    val = str(r.value)
    assert "2" in val and "3" in val


def test_symbolic_diff() -> None:
    from mathmind.math.symbolic import symbolic_diff
    r = symbolic_diff("x**2")
    assert r.error is None
    assert "2*x" in str(r.value) or "2x" in str(r.value)


def test_symbolic_integrate() -> None:
    from mathmind.math.symbolic import symbolic_integrate
    r = symbolic_integrate("2*x", "x")
    assert r.error is None
    assert "x**2" in str(r.value) or "x^2" in str(r.value)


def test_symbolic_simplify() -> None:
    from mathmind.math.symbolic import symbolic_simplify
    r = symbolic_simplify("x + x")
    assert r.error is None
    assert "2*x" in str(r.value) or "2x" in str(r.value)


def test_symbolic_factor() -> None:
    from mathmind.math.symbolic import symbolic_factor
    r = symbolic_factor("x**2 - 1")
    assert r.error is None
    val = str(r.value)
    assert "x" in val


# ---- matrix ----
def test_mat_det_2x2() -> None:
    import numpy as np

    from mathmind.math.matrix import mat_det
    m = np.array([[1, 2], [3, 4]], dtype=float)
    r = mat_det(m)
    assert r.value == pytest.approx(-2.0)


def test_mat_det_identity() -> None:
    import numpy as np

    from mathmind.math.matrix import mat_det
    m = np.eye(3)
    r = mat_det(m)
    assert r.value == pytest.approx(1.0)


def test_mat_inverse() -> None:
    import numpy as np

    from mathmind.math.matrix import mat_inverse
    m = np.array([[4, 7], [2, 6]], dtype=float)
    r = mat_inverse(m)
    assert r.error is None
    result = r.value
    product = m @ result
    assert product == pytest.approx(np.eye(2), abs=1e-9)


def test_mat_singular() -> None:
    import numpy as np

    from mathmind.core.errors import MatrixError
    from mathmind.math.matrix import mat_inverse
    m = np.array([[1, 2], [2, 4]], dtype=float)
    with pytest.raises(MatrixError):
        mat_inverse(m)


def test_mat_transpose() -> None:
    import numpy as np

    from mathmind.math.matrix import mat_ops
    m = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    r = mat_ops(m, "transpose")
    assert r.value.shape == (3, 2)


# ---- stats ----
def test_stats_basic() -> None:
    from mathmind.math.matrix import stats_summary
    r = stats_summary([2, 4, 6])
    assert isinstance(r.value, dict)
    assert r.value["mean"] == pytest.approx(4.0)
    assert r.value["median"] == pytest.approx(4.0)


def test_stats_std() -> None:
    from mathmind.math.matrix import stats_summary
    r = stats_summary([1, 2, 3, 4, 5])
    assert r.value["std"] == pytest.approx(1.5811, rel=1e-3)


def test_linear_regression() -> None:
    from mathmind.math.matrix import linear_regression
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    r = linear_regression(x, y)
    assert r.value["slope"] == pytest.approx(2.0, rel=1e-6)
    assert r.value["r2"] == pytest.approx(1.0, rel=1e-6)


# ---- NL translator ----
def test_nl_percentage() -> None:
    from mathmind.inputs.nl import translate
    expr, translated = translate("15% of 240")
    assert translated
    assert "240" in expr and "15" in expr


def test_nl_percent_off() -> None:
    from mathmind.core.engine import evaluate
    from mathmind.core.symbols import SymbolTable
    from mathmind.inputs.nl import translate
    expr, translated = translate("20% off 2400")
    assert translated
    result = evaluate(expr, SymbolTable())
    assert result == pytest.approx(1920.0)


def test_nl_sqrt() -> None:
    from mathmind.inputs.nl import translate
    expr, translated = translate("square root of 81")
    assert translated
    assert "sqrt" in expr.lower()


def test_nl_squared() -> None:
    from mathmind.inputs.nl import translate
    expr, translated = translate("5 squared")
    assert translated
    assert "**2" in expr or "^2" in expr


def test_nl_no_match() -> None:
    from mathmind.inputs.nl import translate
    _expr, translated = translate("hello world")
    assert not translated


# ---- voice normalizer ----
def test_voice_normalize_squared() -> None:
    from mathmind.inputs.voice import normalize_transcript
    t = normalize_transcript("x squared minus nine equals zero")
    assert "**2" in t or "^2" in t


def test_voice_normalize_numbers() -> None:
    from mathmind.inputs.voice import normalize_transcript
    t = normalize_transcript("mean of two four six ten")
    assert "2" in t and "4" in t


# ---- strategy predictor ----
def test_strategy_predict_returns_tuple() -> None:
    from mathmind.ml.strategy.predict import predict
    strategy, conf, _alts = predict("2 + 3")
    assert isinstance(strategy, str)
    assert 0.0 <= conf <= 1.0


def test_strategy_matrix() -> None:
    from mathmind.ml.strategy.predict import predict
    strategy, _conf, _alts = predict("[[1,2],[3,4]]")
    assert strategy == "matrix-inverse"


# ---- LaTeX renderer ----
def test_latex_fraction() -> None:
    from mathmind.core.parser import parse
    from mathmind.export.latex import ast_to_latex
    node = parse("1/2")
    latex = ast_to_latex(node)
    assert r"\frac" in latex


def test_latex_power() -> None:
    from mathmind.core.parser import parse
    from mathmind.export.latex import ast_to_latex
    node = parse("x**2")
    latex = ast_to_latex(node)
    assert "^" in latex


def test_latex_sqrt() -> None:
    from mathmind.core.parser import parse
    from mathmind.export.latex import ast_to_latex
    node = parse("sqrt(4)")
    latex = ast_to_latex(node)
    assert r"\sqrt" in latex


# ---- storage ----
def test_store_save_and_retrieve(tmp_path) -> None:

    from mathmind.models import Result, Step
    from mathmind.store import get_history, save_result

    db = tmp_path / "test.db"
    r = Result(value=42.0, steps=[Step(1, "Test", "2+2", "4")], expression="2+2", latex="2+2")
    save_result(r, raw_input="2+2", db_path=db)
    rows = get_history(5, db_path=db)
    assert len(rows) == 1
    assert rows[0]["expression"] == "2+2"


# ---- vision fallback ----
def test_vision_fallback_predict() -> None:
    from mathmind.inputs.vision.infer import predict_symbols_fallback, predictions_to_expression
    from mathmind.inputs.vision.train_fallback import _render_sample

    crops = [_render_sample("1"), _render_sample("+"), _render_sample("2")]
    preds = predict_symbols_fallback(crops)
    assert len(preds) == 3
    expr, conf = predictions_to_expression(preds)
    assert len(expr) == 3
    assert conf > 0.0
