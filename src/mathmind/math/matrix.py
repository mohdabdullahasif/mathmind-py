"""Matrix and statistics operations for MathMind."""
from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from mathmind.core.errors import MatrixError
from mathmind.models import Result, Step


# ------------------------------------------------------------------ matrix ops
def mat_det(matrix: np.ndarray) -> Result:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise MatrixError("Determinant requires a square matrix")
    n = matrix.shape[0]
    det = float(np.linalg.det(matrix))
    steps = _det_steps(matrix, n, det)
    return Result(value=det, steps=steps, strategy="matrix-det", expression=str(matrix.tolist()))


def _det_steps(m: np.ndarray, n: int, det: float) -> list[Step]:
    steps = [Step(1, "Matrix input", str(m.tolist()), str(m.tolist()))]
    if n == 2:
        a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
        steps.append(Step(2, "2×2 det = ad - bc", f"({a})({d}) - ({b})({c})", str(det)))
    else:
        steps.append(Step(2, f"{n}×{n} determinant via LU decomposition", "", str(det)))
    return steps


def mat_inverse(matrix: np.ndarray) -> Result:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise MatrixError("Inverse requires a square matrix")
    det = float(np.linalg.det(matrix))
    if abs(det) < 1e-12:
        raise MatrixError("Matrix is singular (det ≈ 0)")
    inv = np.linalg.inv(matrix)
    n = matrix.shape[0]
    steps = _det_steps(matrix, n, det)
    steps.append(Step(len(steps) + 1, "Compute inverse via adjugate / det", "", str(inv.tolist())))
    return Result(value=inv, steps=steps, strategy="matrix-inverse", expression=str(matrix.tolist()))


def mat_ops(matrix: np.ndarray, op: str) -> Result:
    """Transpose, rank, trace, eigenvalues, eigenvectors."""
    match op:
        case "transpose":
            transposed = matrix.T
            return Result(value=transposed, steps=[Step(1, "Transpose matrix", str(matrix.tolist()), str(transposed.tolist()))], strategy="matrix-transpose")
        case "rank":
            rank_val = int(np.linalg.matrix_rank(matrix))
            return Result(value=rank_val, steps=[Step(1, "Compute rank", str(matrix.tolist()), str(rank_val))], strategy="matrix-rank")
        case "trace":
            trace_val = float(np.trace(matrix))
            return Result(value=trace_val, steps=[Step(1, "Trace = sum of diagonal", str(matrix.tolist()), str(trace_val))], strategy="matrix-trace")
        case "eigenvalues":
            vals, _vecs = np.linalg.eig(matrix)
            return Result(value=vals.tolist(), steps=[Step(1, "Compute eigenvalues", str(matrix.tolist()), str(vals.tolist()))], strategy="eigenvalues")
        case _:
            raise MatrixError(f"Unknown matrix operation: {op}")


def mat_solve(A: np.ndarray, b: np.ndarray) -> Result:
    """Solve Ax = b."""
    if abs(float(np.linalg.det(A))) < 1e-12:
        raise MatrixError("Matrix A is singular; system may have no unique solution")
    x = np.linalg.solve(A, b)
    steps = [
        Step(1, "Set up Ax = b", f"A={A.tolist()}, b={b.tolist()}", ""),
        Step(2, "Solve via LU decomposition", "", str(x.tolist())),
    ]
    return Result(value=x.tolist(), steps=steps, strategy="matrix-inverse", expression=f"A={A.tolist()}")


# ------------------------------------------------------------------ statistics
def stats_summary(data: list[float]) -> Result:
    arr = np.array(data)
    n = len(arr)
    if n == 0:
        raise ValueError("Empty dataset")

    mean_v = float(np.mean(arr))
    median_v = float(np.median(arr))
    std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    var_v = float(np.var(arr, ddof=1)) if n > 1 else 0.0
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    # mode
    mode_res = scipy_stats.mode(arr, keepdims=True)
    mode_v = float(mode_res.mode[0])

    summary = {
        "n": n,
        "mean": mean_v,
        "median": median_v,
        "mode": mode_v,
        "std": std_v,
        "variance": var_v,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }
    steps = [
        Step(1, f"Dataset: {data}", "", ""),
        Step(2, f"Mean = {mean_v:.4f}", "", ""),
        Step(3, f"Median = {median_v}", "", ""),
        Step(4, f"Std Dev = {std_v:.4f}", "", ""),
        Step(5, f"Q1={q1}, Q3={q3}, IQR={q3-q1}", "", ""),
    ]
    return Result(value=summary, steps=steps, strategy="statistics", expression=str(data))


def linear_regression(x: list[float], y: list[float]) -> Result:
    xarr = np.array(x)
    yarr = np.array(y)
    slope, intercept, r_value, p_value, _std_err = scipy_stats.linregress(xarr, yarr)
    r2 = r_value ** 2
    steps = [
        Step(1, "Compute means", "", f"x̄={float(np.mean(xarr)):.4f}, ȳ={float(np.mean(yarr)):.4f}"),
        Step(2, "Compute slope m", "", f"m = {slope:.4f}"),
        Step(3, "Compute intercept b", "", f"b = {intercept:.4f}"),
        Step(4, "R² score", "", f"R² = {r2:.4f}"),
    ]
    return Result(
        value={"slope": slope, "intercept": intercept, "r2": r2, "p_value": p_value},
        steps=steps,
        strategy="linear-regression",
        expression=f"x={x}, y={y}",
    )
