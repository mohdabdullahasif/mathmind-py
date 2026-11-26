"""Typer CLI entry point for MathMind."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mathmind.core.engine import solve
from mathmind.core.symbols import SymbolTable
from mathmind.inputs.nl import translate
from mathmind.inputs.voice import normalize_transcript
from mathmind.math.symbolic import (
    symbolic_diff,
    symbolic_expand,
    symbolic_factor,
    symbolic_integrate,
    symbolic_simplify,
    symbolic_solve,
)
from mathmind.math.matrix import mat_det, mat_inverse, mat_ops, stats_summary
from mathmind.ml.strategy.predict import predict as predict_strategy
from mathmind.store import get_history, save_result

app = typer.Typer(name="mathmind", help="MathMind – AI-powered math assistant", rich_markup_mode="rich")
console = Console()

_global_symbols = SymbolTable()


def _display_result(result, label: str = "Result") -> None:
    color = "red" if result.error else "green"
    value = result.error if result.error else str(result.value)
    console.print(Panel(f"[bold]{value}[/bold]", title=f"[{color}]{label}[/{color}]", border_style=color))

    if result.strategy and not result.error:
        console.print(f"[dim]Strategy: {result.strategy}  Confidence: {result.confidence:.0%}[/dim]")

    if result.steps and not result.error:
        t = Table("Step", "Description", "Result", show_header=True)
        for s in result.steps:
            t.add_row(str(s.step_no), s.description, s.after or s.before)
        console.print(t)

    if result.latex and not result.error:
        console.print(f"[dim]LaTeX: {result.latex}[/dim]")


@app.command("solve")
def cmd_solve(
    expression: str = typer.Argument(..., help="Math expression or equation"),
    var: str = typer.Option("x", "--var", "-v", help="Variable for solving"),
    steps: bool = typer.Option(True, "--steps/--no-steps", help="Show solution steps"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Export PDF path"),
) -> None:
    """Solve or evaluate a math expression."""
    strategy, confidence, alts = predict_strategy(expression)
    result = solve(expression, _global_symbols, strategy=strategy, confidence=confidence)
    _display_result(result)
    if alts:
        console.print(f"[dim]Alternatives: {', '.join(f'{s} ({c:.0%})' for s, c in alts)}[/dim]")
    save_result(result, raw_input=expression)
    if export:
        from mathmind.export.pdf import export_pdf
        path = export_pdf(result, export)
        console.print(f"[blue]Exported to {path}[/blue]")


@app.command("symbolic")
def cmd_symbolic(
    expression: str = typer.Argument(..., help="Expression or equation"),
    op: str = typer.Option("solve", "--op", "-o", help="Operation: solve|diff|integrate|simplify|expand|factor"),
    var: str = typer.Option("x", "--var", "-v"),
    lower: Optional[str] = typer.Option(None, "--lower"),
    upper: Optional[str] = typer.Option(None, "--upper"),
) -> None:
    """Symbolic math operations (solve, diff, integrate, simplify, expand, factor)."""
    match op:
        case "solve":
            result = symbolic_solve(expression, var)
        case "diff":
            result = symbolic_diff(expression, var)
        case "integrate":
            result = symbolic_integrate(expression, var, lower, upper)
        case "simplify":
            result = symbolic_simplify(expression)
        case "expand":
            result = symbolic_expand(expression)
        case "factor":
            result = symbolic_factor(expression)
        case _:
            console.print(f"[red]Unknown operation: {op}[/red]")
            raise typer.Exit(1)
    _display_result(result, label=op.capitalize())
    save_result(result, raw_input=expression)


@app.command("nl")
def cmd_nl(
    text: str = typer.Argument(..., help="Natural language math problem"),
) -> None:
    """Translate natural language to math and solve it."""
    expression, was_translated = translate(text)
    if was_translated:
        console.print(f"[yellow]Translated: {expression}[/yellow]")
    else:
        console.print(f"[dim]No NL pattern matched. Trying expression directly.[/dim]")
    result = solve(expression, _global_symbols, channel="nl")
    _display_result(result)
    save_result(result, raw_input=text)


@app.command("matrix")
def cmd_matrix(
    expression: str = typer.Argument(..., help="Matrix literal, e.g. [[1,2],[3,4]]"),
    op: str = typer.Option("det", "--op", "-o", help="det|inverse|transpose|rank|trace|eigenvalues"),
) -> None:
    """Matrix operations."""
    import numpy as np
    from mathmind.core.engine import solve as eng_solve
    r = eng_solve(expression, _global_symbols)
    if r.error or not isinstance(r.value, np.ndarray):
        console.print(f"[red]Could not parse matrix: {r.error or 'not a matrix'}[/red]")
        raise typer.Exit(1)
    matrix = r.value
    match op:
        case "det":
            result = mat_det(matrix)
        case "inverse":
            result = mat_inverse(matrix)
        case _:
            result = mat_ops(matrix, op)
    _display_result(result, label=op)
    save_result(result, raw_input=expression)


@app.command("stats")
def cmd_stats(
    data: str = typer.Argument(..., help="Comma-separated numbers, e.g. 1,2,3,4,5"),
) -> None:
    """Descriptive statistics on a list of numbers."""
    try:
        numbers = [float(x.strip()) for x in data.split(",")]
    except ValueError:
        console.print("[red]Invalid data. Use comma-separated numbers.[/red]")
        raise typer.Exit(1)
    result = stats_summary(numbers)
    if isinstance(result.value, dict):
        t = Table("Statistic", "Value", show_header=True)
        for k, v in result.value.items():
            t.add_row(k, f"{v:.4g}")
        console.print(t)
    save_result(result, raw_input=data)


@app.command("repl")
def cmd_repl() -> None:
    """Interactive REPL with history and tab completion."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.styles import Style

        style = Style.from_dict({"prompt": "ansicyan bold"})
        session: PromptSession = PromptSession(
            history=InMemoryHistory(),
            completer=WordCompleter(_global_symbols.all_names(), ignore_case=True),
            style=style,
        )
    except ImportError:
        session = None  # type: ignore[assignment]

    console.print(Panel("[bold cyan]MathMind REPL[/bold cyan]\nType an expression, 'quit' to exit, 'help' for commands.", border_style="cyan"))

    while True:
        try:
            if session:
                text = session.prompt(">>> ", style=style if session else None)
            else:
                text = input(">>> ")
        except (KeyboardInterrupt, EOFError):
            break

        text = text.strip()
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            break
        if text.lower() == "help":
            console.print("[cyan]Commands:[/cyan] solve, symbolic, nl, matrix, stats, history, quit")
            continue
        if text.lower() == "history":
            rows = get_history(10)
            t = Table("Time", "Channel", "Expression", "Result")
            for r in rows:
                t.add_row(r["created_at"][:19], r["channel"], r["expression"][:40], str(r["result"])[:30])
            console.print(t)
            continue

        # natural-language detection
        if any(c.isalpha() for c in text) and not any(c in text for c in "+-*/=^"):
            expr, translated = translate(text)
            if translated:
                console.print(f"[yellow]NL→ {expr}[/yellow]")
                text = expr

        strategy, confidence, alts = predict_strategy(text)
        result = solve(text, _global_symbols, strategy=strategy, confidence=confidence)
        _display_result(result)
        if not result.error:
            save_result(result, raw_input=text)


@app.command("history")
def cmd_history(limit: int = typer.Option(20, "--limit", "-n")) -> None:
    """Show solve history."""
    rows = get_history(limit)
    if not rows:
        console.print("[yellow]No history yet.[/yellow]")
        return
    t = Table("ID", "Time", "Channel", "Expression", "Result", "Strategy")
    for r in rows:
        t.add_row(str(r["id"]), r["created_at"][:19], r["channel"], r["expression"][:40], str(r["result"])[:25], r["strategy"] or "")
    console.print(t)


@app.command("export")
def cmd_export(
    expression: str = typer.Argument(..., help="Expression to solve and export"),
    output: str = typer.Option("output.pdf", "--output", "-o"),
    title: str = typer.Option("MathMind Solution", "--title"),
) -> None:
    """Solve an expression and export to PDF."""
    result = solve(expression, _global_symbols)
    from mathmind.export.pdf import export_pdf
    path = export_pdf(result, output, title=title)
    console.print(f"[green]Exported to {path}[/green]")


@app.command("train")
def cmd_train(
    n: int = typer.Option(10000, "--n", help="Number of training samples"),
    save_dir: str = typer.Option("models", "--save-dir"),
) -> None:
    """Train the strategy prediction ML model."""
    console.print(f"[yellow]Training on {n} samples...[/yellow]")
    from mathmind.ml.strategy.train import train_and_evaluate
    results = train_and_evaluate(n, save_dir)
    console.print(f"[green]Best model: {results['best_model']} (macro F1: {results['best_f1']:.4f})[/green]")
    t = Table("Model", "Macro F1")
    for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
        if name in results:
            t.add_row(name, f"{results[name]['macro_f1']:.4f}")
    console.print(t)


if __name__ == "__main__":
    app()
