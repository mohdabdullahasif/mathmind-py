"""Strategy ML feature extractor for problem classification."""
from __future__ import annotations

import re

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


def extract_features(problem: str) -> np.ndarray:
    """Extract a 14-element numeric feature vector from a math problem string."""
    text = problem.lower().strip()

    is_eq = 1.0 if "=" in text else 0.0
    is_matrix = 1.0 if "[[" in text else 0.0
    has_trig = 1.0 if any(f in text for f in TRIG_FUNS) else 0.0
    has_log = 1.0 if any(f in text for f in LOG_FUNS) else 0.0
    has_diff = 1.0 if re.search(r"\b(diff|differentiate|derivative)\b", text) else 0.0
    has_integrate = 1.0 if re.search(r"\b(integrate|integral)\b", text) else 0.0
    has_semicolon = 1.0 if ";" in text else 0.0
    has_isolated_var = (
        1.0 if re.search(r"(^|;)\s*[a-z]\s*=", text) else 0.0
    )

    var_names = set(re.findall(r"\b[a-z]\b", text)) - {"e"}
    num_vars = float(min(len(var_names), 5))

    max_degree = 0.0
    is_linear = 0.0
    is_quadratic = 0.0
    if not (has_diff or has_integrate or is_matrix or has_semicolon):
        try:
            lhs = problem.split("=")[0]
            for sym_name in list(var_names)[:1]:
                sym = sp.Symbol(sym_name)
                parsed = sp.sympify(lhs, locals={sym_name: sym})
                poly = sp.Poly(parsed, sym)
                d = int(sp.degree(poly)) if poly.is_univariate else 0
                max_degree = float(d)
                is_linear = 1.0 if d == 1 else 0.0
                is_quadratic = 1.0 if d == 2 else 0.0
                break
        except (sp.SympifyError, ValueError, TypeError):
            pass

    n_ops = float(len(re.findall(r"[+\-*/^]", problem)))
    has_parens = 1.0 if "(" in problem else 0.0

    return np.array(
        [
            is_eq,
            is_matrix,
            has_trig,
            has_log,
            has_diff,
            has_integrate,
            has_semicolon,
            has_isolated_var,
            num_vars,
            max_degree,
            is_linear,
            is_quadratic,
            n_ops,
            has_parens,
        ],
        dtype=float,
    )
