"""AST node definitions for MathMind expression engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass
class NumberNode:
    value: float
    col: int = 0


@dataclass
class IdentNode:
    name: str
    col: int = 0


@dataclass
class UnaryNode:
    op: str  # '+' or '-'
    operand: "ASTNode"
    col: int = 0


@dataclass
class BinaryNode:
    op: str  # '+' '-' '*' '/' '%' '**'
    left: "ASTNode"
    right: "ASTNode"
    col: int = 0


@dataclass
class CallNode:
    name: str
    args: list["ASTNode"]
    col: int = 0


@dataclass
class AssignNode:
    name: str
    params: list[str]  # empty for variable, non-empty for function def
    body: "ASTNode"
    col: int = 0


@dataclass
class MatrixNode:
    rows: list[list["ASTNode"]]
    col: int = 0


ASTNode = Union[
    NumberNode, IdentNode, UnaryNode, BinaryNode, CallNode, AssignNode, MatrixNode
]
