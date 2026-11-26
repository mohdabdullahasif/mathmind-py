"""Strategy ML feature extractor.

Extracts features from an AST for strategy prediction:
- max polynomial degree
- number of variables
- presence of trig/log functions
- is equation vs expression
- matrix present
- node counts
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import sympy as sp


STRATEGY_LABELS = [
    "direct-arithmetic",
    "linear-isolate",
    "quadratic-formula",
    "factoring",
    "substitution",
    "elimination",
    "matrix-inverse",
    "differentiate",
    "integrate",
]

TRIG_FUNS = {"sin", "cos", "tan", "asin", "acos", "atan"}
LOG_FUNS = {"log", "ln", "log10", "log2"}
CALC_FUNS = {"diff", "differentiate", "integrate", "integral"}


def extract_features(problem: str) -> np.ndarray:
    """Extract a feature vector from a problem string.

    Returns a 1-D float array of length 12.
    """
    text = problem.lower()

    # 1. has equation (=)
    is_eq = 1.0 if "=" in text else 0.0

    # 2. matrix present
    is_matrix = 1.0 if "[[" in text else 0.0

    # 3. trig functions present
    has_trig = 1.0 if any(f in text for f in TRIG_FUNS) else 0.0

    # 4. log functions present
    has_log = 1.0 if any(f in text for f in LOG_FUNS) else 0.0

    # 5. calculus keywords
    has_calc = 1.0 if any(f in text for f in CALC_FUNS) else 0.0

    # 6. number of distinct variable names (single-letter identifiers)
    var_names = set(re.findall(r"\b[a-z]\b", text)) - {"e"}
    num_vars = float(min(len(var_names), 5))

    # 7-8. try sympy degree analysis
    max_degree = 0.0
    is_linear = 0.0
    is_quadratic = 0.0
    try:
        expr_str = text.split("=")[0] if "=" in text else text
        expr_str = re.sub(r"\*\*", "^", expr_str)
        for sym_name in list(var_names)[:1]:
            sym = sp.Symbol(sym_name)
            parsed = sp.sympify(problem.split("=")[0] if "=" in problem else problem,
                                locals={sym_name: sym})
            d = sp.degree(sp.Poly(parsed, sym)) if sp.Poly(parsed, sym).is_univariate else 0
            max_degree = float(d)
            is_linear = 1.0 if d == 1 else 0.0
            is_quadratic = 1.0 if d == 2 else 0.0
            break
    except Exception:
        pass

    # 9. expression length bucket (0-short, 1-medium, 2-long)
    length_bucket = min(len(problem) // 20, 2.0)

    # 10. number of operators
    n_ops = float(len(re.findall(r"[+\-*/^]", problem)))

    # 11. has parentheses
    has_parens = 1.0 if "(" in problem else 0.0

    # 12. contains "solve" / "find" keyword
    has_solve_kw = 1.0 if re.search(r"\b(solve|find|calculate|compute)\b", text) else 0.0

    return np.array([
        is_eq, is_matrix, has_trig, has_log, has_calc,
        num_vars, max_degree, is_linear, is_quadratic,
        length_bucket, n_ops, has_parens,
    ], dtype=float)
