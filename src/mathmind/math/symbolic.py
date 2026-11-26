"""Symbolic math module using SymPy + own step generation."""
from __future__ import annotations

import sympy as sp

from mathmind.models import Result, Step


def _parse_sympy(expr_str: str) -> sp.Expr:
    """Parse *expr_str* with sympy's parser (safe, not eval)."""
    return sp.sympify(expr_str, evaluate=True)


def _steps_simplify(expr: sp.Expr) -> list[Step]:
    simplified = sp.simplify(expr)
    return [
        Step(1, "Original expression", str(expr), str(expr)),
        Step(2, "Apply simplification rules", str(expr), str(simplified)),
    ]


def _steps_expand(expr: sp.Expr) -> list[Step]:
    expanded = sp.expand(expr)
    return [
        Step(1, "Original expression", str(expr), str(expr)),
        Step(2, "Expand using distributive law", str(expr), str(expanded)),
    ]


def _steps_factor(expr: sp.Expr) -> list[Step]:
    factored = sp.factor(expr)
    return [
        Step(1, "Original expression", str(expr), str(expr)),
        Step(2, "Factor expression", str(expr), str(factored)),
    ]


def symbolic_simplify(expr_str: str) -> Result:
    expr = _parse_sympy(expr_str)
    result = sp.simplify(expr)
    steps = _steps_simplify(expr)
    return Result(
        value=str(result),
        steps=steps,
        latex=sp.latex(result),
        strategy="simplify",
        expression=expr_str,
    )


def symbolic_expand(expr_str: str) -> Result:
    expr = _parse_sympy(expr_str)
    result = sp.expand(expr)
    steps = _steps_expand(expr)
    return Result(
        value=str(result),
        steps=steps,
        latex=sp.latex(result),
        strategy="expand",
        expression=expr_str,
    )


def symbolic_factor(expr_str: str) -> Result:
    expr = _parse_sympy(expr_str)
    result = sp.factor(expr)
    steps = _steps_factor(expr)
    return Result(
        value=str(result),
        steps=steps,
        latex=sp.latex(result),
        strategy="factor",
        expression=expr_str,
    )


def symbolic_diff(expr_str: str, var: str = "x") -> Result:
    sym = sp.Symbol(var)
    expr = _parse_sympy(expr_str)
    derivative = sp.diff(expr, sym)
    steps = [
        Step(1, f"Differentiate with respect to {var}", str(expr), str(expr)),
        Step(2, "Apply differentiation rules", str(expr), str(derivative)),
    ]
    return Result(
        value=str(derivative),
        steps=steps,
        latex=sp.latex(derivative),
        strategy="differentiate",
        expression=expr_str,
    )


def symbolic_integrate(
    expr_str: str, var: str = "x", lower: str | None = None, upper: str | None = None
) -> Result:
    sym = sp.Symbol(var)
    expr = _parse_sympy(expr_str)
    if lower is not None and upper is not None:
        lb = _parse_sympy(lower)
        ub = _parse_sympy(upper)
        result = sp.integrate(expr, (sym, lb, ub))
        desc = f"Definite integral from {lower} to {upper}"
    else:
        result = sp.integrate(expr, sym)
        desc = "Indefinite integral (+ C)"
    steps = [
        Step(1, "Expression to integrate", str(expr), str(expr)),
        Step(2, desc, str(expr), str(result)),
    ]
    return Result(
        value=str(result),
        steps=steps,
        latex=sp.latex(result),
        strategy="integrate",
        expression=expr_str,
    )


def symbolic_solve(equation_str: str, var: str = "x") -> Result:
    """Solve an equation (may contain '=') for *var*."""
    sym = sp.Symbol(var)
    if "=" in equation_str:
        left_str, right_str = equation_str.split("=", 1)
        lhs = _parse_sympy(left_str.strip())
        rhs = _parse_sympy(right_str.strip())
        expr = lhs - rhs
    else:
        expr = _parse_sympy(equation_str)

    solutions = sp.solve(expr, sym)
    steps = _build_solve_steps(equation_str, expr, solutions, var)
    sol_str = ", ".join(f"{var} = {s}" for s in solutions) if solutions else "No solution"
    return Result(
        value=sol_str,
        steps=steps,
        latex=sp.latex(solutions),
        strategy=_detect_strategy(expr, sym),
        expression=equation_str,
    )


def _detect_strategy(expr: sp.Expr, sym: sp.Symbol) -> str:
    degree = sp.degree(expr, sym)
    if degree == 1:
        return "linear-isolate"
    if degree == 2:
        discriminant = sp.discriminant(expr, sym)
        if sp.simplify(discriminant) >= 0:
            return "quadratic-formula"
        return "quadratic-formula"
    return "polynomial"


def _build_solve_steps(
    orig: str, expr: sp.Expr, solutions: list, var: str
) -> list[Step]:
    steps = [Step(1, "Original equation", orig, orig)]
    if "=" in orig:
        steps.append(Step(2, "Move all terms to left side (LHS - RHS = 0)", orig, f"{expr} = 0"))
    degree = sp.degree(expr, sp.Symbol(var))
    if degree == 1:
        steps.append(Step(3, f"Isolate {var}", f"{expr} = 0", str(solutions[0]) if solutions else ""))
    elif degree == 2:
        a, b, c = sp.Poly(expr, sp.Symbol(var)).all_coeffs()
        disc = b**2 - 4*a*c
        steps.append(Step(3, "Identify coefficients a, b, c", str(expr), f"a={a}, b={b}, c={c}"))
        steps.append(Step(4, "Discriminant Δ = b²-4ac", "", str(disc)))
        steps.append(Step(5, "Apply quadratic formula", "", str(solutions)))
    sol_str = ", ".join(f"{var} = {s}" for s in solutions)
    steps.append(Step(len(steps) + 1, "Solution", "", sol_str))
    return steps
