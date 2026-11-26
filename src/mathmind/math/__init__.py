"""__init__ for math package."""
from mathmind.math.matrix import (
    linear_regression,
    mat_det,
    mat_inverse,
    mat_ops,
    mat_solve,
    stats_summary,
)
from mathmind.math.symbolic import (
    symbolic_diff,
    symbolic_expand,
    symbolic_factor,
    symbolic_integrate,
    symbolic_simplify,
    symbolic_solve,
)

__all__ = [
    "symbolic_simplify",
    "symbolic_expand",
    "symbolic_factor",
    "symbolic_diff",
    "symbolic_integrate",
    "symbolic_solve",
    "mat_det",
    "mat_inverse",
    "mat_ops",
    "mat_solve",
    "stats_summary",
    "linear_regression",
]
