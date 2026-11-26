"""LaTeX renderer: converts AST nodes to LaTeX strings."""
from __future__ import annotations

from mathmind.core.ast import (
    AssignNode,
    ASTNode,
    BinaryNode,
    CallNode,
    IdentNode,
    MatrixNode,
    NumberNode,
    UnaryNode,
)

_LATEX_FUNS = {
    "sqrt": r"\sqrt",
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "asin": r"\arcsin",
    "acos": r"\arccos",
    "atan": r"\arctan",
    "log": r"\log",
    "ln": r"\ln",
    "abs": r"\left|",
}


def ast_to_latex(node: ASTNode) -> str:
    """Render an AST node as a LaTeX string."""
    match node:
        case NumberNode(value=v):
            return str(int(v)) if v == int(v) else str(v)
        case IdentNode(name=n):
            return n
        case UnaryNode(op=op, operand=operand):
            return f"{op}{ast_to_latex(operand)}"
        case BinaryNode(op="/", left=left, right=right):
            return rf"\frac{{{ast_to_latex(left)}}}{{{ast_to_latex(right)}}}"
        case BinaryNode(op="**", left=left, right=right):
            return rf"{{{ast_to_latex(left)}}}^{{{ast_to_latex(right)}}}"
        case BinaryNode(op=op, left=left, right=right):
            sym = r"\cdot" if op == "*" else op
            return f"{ast_to_latex(left)} {sym} {ast_to_latex(right)}"
        case CallNode(name="sqrt", args=[arg]):
            return rf"\sqrt{{{ast_to_latex(arg)}}}"
        case CallNode(name="abs", args=[arg]):
            return rf"\left|{ast_to_latex(arg)}\right|"
        case CallNode(name=name, args=args):
            fn = _LATEX_FUNS.get(name, f"\\operatorname{{{name}}}")
            inner = ", ".join(ast_to_latex(a) for a in args)
            return rf"{fn}\left({inner}\right)"
        case AssignNode(name=name, params=params, body=body):
            if params:
                p = ", ".join(params)
                return rf"{name}({p}) = {ast_to_latex(body)}"
            return rf"{name} = {ast_to_latex(body)}"
        case MatrixNode(rows=rows):
            inner = r" \\ ".join(
                " & ".join(ast_to_latex(cell) for cell in row) for row in rows
            )
            return rf"\begin{{pmatrix}} {inner} \end{{pmatrix}}"
        case _:
            return str(node)
