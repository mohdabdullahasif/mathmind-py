"""Own rule-based simplifier on the AST.

Rules: constant folding, x+0, x*1, x*0, x-x, x**1, x**0 etc.
"""
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


def _is_zero(node: ASTNode) -> bool:
    return isinstance(node, NumberNode) and node.value == 0.0


def _is_one(node: ASTNode) -> bool:
    return isinstance(node, NumberNode) and node.value == 1.0


def _same_ident(a: ASTNode, b: ASTNode) -> bool:
    return isinstance(a, IdentNode) and isinstance(b, IdentNode) and a.name == b.name


def simplify(node: ASTNode) -> ASTNode:
    """Recursively apply algebraic identities to *node*."""
    match node:
        case NumberNode():
            return node
        case IdentNode():
            return node
        case UnaryNode(op=op, operand=operand):
            s = simplify(operand)
            if isinstance(s, NumberNode):
                return NumberNode(-s.value if op == "-" else s.value)
            return UnaryNode(op, s)
        case BinaryNode(op=op, left=left, right=right):
            sl = simplify(left)
            sr = simplify(right)
            # constant folding
            if isinstance(sl, NumberNode) and isinstance(sr, NumberNode):
                if op == "+":
                    return NumberNode(sl.value + sr.value)
                if op == "-":
                    return NumberNode(sl.value - sr.value)
                if op == "*":
                    return NumberNode(sl.value * sr.value)
                if op == "/" and sr.value != 0:
                    return NumberNode(sl.value / sr.value)
                if op == "**":
                    try:
                        return NumberNode(sl.value ** sr.value)
                    except (OverflowError, ValueError):
                        pass
            # algebraic identities
            match op:
                case "+":
                    if _is_zero(sl): return sr
                    if _is_zero(sr): return sl
                case "-":
                    if _is_zero(sr): return sl
                    if _same_ident(sl, sr): return NumberNode(0.0)
                case "*":
                    if _is_zero(sl) or _is_zero(sr): return NumberNode(0.0)
                    if _is_one(sl): return sr
                    if _is_one(sr): return sl
                case "**":
                    if _is_zero(sr): return NumberNode(1.0)
                    if _is_one(sr): return sl
                    if _is_zero(sl): return NumberNode(0.0)
            return BinaryNode(op, sl, sr)
        case CallNode(name=name, args=args):
            return CallNode(name, [simplify(a) for a in args])
        case AssignNode(name=name, params=params, body=body):
            return AssignNode(name, params, simplify(body))
        case MatrixNode(rows=rows):
            return MatrixNode([[simplify(c) for c in row] for row in rows])
        case _:
            return node
