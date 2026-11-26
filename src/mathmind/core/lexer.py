"""Lexer for MathMind expression engine.

Tokenizes: numbers (int, float, sci), operators, parens, identifiers,
           =, commas, brackets.
No eval() or exec() used anywhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto

from mathmind.core.errors import LexError


class TT(Enum):
    """Token type enum."""
    NUMBER = auto()
    IDENT = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    STARSTAR = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    EQUALS = auto()
    SEMICOLON = auto()
    EOF = auto()


@dataclass
class Token:
    type: TT
    value: str
    col: int


_NUMBER_RE = re.compile(r"\d+(\.\d*)?([eE][+-]?\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_OP_MAP: dict[str, TT] = {
    "+": TT.PLUS,
    "-": TT.MINUS,
    "*": TT.STAR,
    "/": TT.SLASH,
    "%": TT.PERCENT,
    "(": TT.LPAREN,
    ")": TT.RPAREN,
    "[": TT.LBRACKET,
    "]": TT.RBRACKET,
    ",": TT.COMMA,
    "=": TT.EQUALS,
    ";": TT.SEMICOLON,
}


def tokenize(text: str) -> list[Token]:
    """Lex *text* into a token list; raises LexError on bad input."""
    tokens: list[Token] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # whitespace
        if ch in " \t\r\n":
            i += 1
            continue

        # ** before *
        if text[i : i + 2] == "**":
            tokens.append(Token(TT.STARSTAR, "**", i))
            i += 2
            continue

        # ^ alias for **
        if ch == "^":
            tokens.append(Token(TT.STARSTAR, "**", i))
            i += 1
            continue

        # number
        m = _NUMBER_RE.match(text, i)
        if m:
            tokens.append(Token(TT.NUMBER, m.group(), i))
            i = m.end()
            continue

        # identifier
        m = _IDENT_RE.match(text, i)
        if m:
            tokens.append(Token(TT.IDENT, m.group(), i))
            i = m.end()
            continue

        # single-char operators
        if ch in _OP_MAP:
            tokens.append(Token(_OP_MAP[ch], ch, i))
            i += 1
            continue

        raise LexError(f"Unexpected character '{ch}'", i)

    tokens.append(Token(TT.EOF, "", n))
    return tokens
