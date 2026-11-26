"""Step generator for MathMind symbolic operations."""
from __future__ import annotations

from mathmind.models import Step


def linear_steps(equation: str, a: float, b: float, var: str = "x") -> list[Step]:
    """Generate steps for ax + b = 0."""
    steps = [
        Step(1, f"Equation: {equation}", equation, equation),
        Step(2, "Move constant to right side", f"{a}{var} + {b} = 0", f"{a}{var} = {-b}"),
        Step(3, f"Divide both sides by {a}", f"{a}{var} = {-b}", f"{var} = {-b / a:.6g}"),
    ]
    return steps


def quadratic_steps(equation: str, a: float, b: float, c: float, solutions: list, var: str = "x") -> list[Step]:
    """Generate steps for ax²+bx+c=0."""
    disc = b ** 2 - 4 * a * c
    steps = [
        Step(1, f"Equation: {equation}", equation, equation),
        Step(2, "Identify a, b, c", equation, f"a={a}, b={b}, c={c}"),
        Step(3, "Discriminant Δ = b²-4ac", "", f"Δ = {b}² - 4({a})({c}) = {disc}"),
        Step(4, "Apply quadratic formula x = (-b ± √Δ) / 2a", "", str(solutions)),
    ]
    return steps
