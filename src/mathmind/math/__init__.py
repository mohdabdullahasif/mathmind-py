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
    "linear_regression",
    "mat_det",
    "mat_inverse",
    "mat_ops",
    "mat_solve",
    "stats_summary",
    "symbolic_diff",
    "symbolic_expand",
    "symbolic_factor",
    "symbolic_integrate",
    "symbolic_simplify",
    "symbolic_solve",
]
