"""Recursive-descent parser for MathMind.

Grammar (EBNF):
  stmt    = IDENT '(' params ')' '=' expr
           | IDENT '=' expr
           | expr ;
  expr    = term { ('+' | '-') term } ;
  term    = unary { ('*' | '/' | '%') unary } ;
  unary   = ['-' | '+'] power ;
  power   = primary [ '**' unary ] ;   # right-associative
  primary = NUMBER
           | IDENT
           | IDENT '(' [args] ')'
           | '(' expr ')'
           | '[' rows ']'              # matrix literal
           | implicit_mul ;

Implicit multiplication: 2x, 3(x+1), x(y) are supported.

No eval() or exec() used anywhere.
"""
from __future__ import annotations

from mathmind.core.ast import (
    ASTNode,
    AssignNode,
    BinaryNode,
    CallNode,
    IdentNode,
    MatrixNode,
    NumberNode,
    UnaryNode,
)
from mathmind.core.errors import ParseError
from mathmind.core.lexer import TT, Token, tokenize


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # ------------------------------------------------------------------ helpers
    @property
    def _cur(self) -> Token:
        return self._tokens[self._pos]

    def _peek(self, offset: int = 1) -> Token:
        idx = min(self._pos + offset, len(self._tokens) - 1)
        return self._tokens[idx]

    def _advance(self) -> Token:
        tok = self._cur
        if tok.type != TT.EOF:
            self._pos += 1
        return tok

    def _expect(self, tt: TT) -> Token:
        if self._cur.type != tt:
            raise ParseError(
                f"Expected {tt.name} but got '{self._cur.value}'", self._cur.col
            )
        return self._advance()

    def _match(self, *types: TT) -> bool:
        return self._cur.type in types

    # ------------------------------------------------------------------ grammar
    def parse_stmt(self) -> ASTNode:
        # function definition: IDENT '(' params ')' '='
        if (
            self._cur.type == TT.IDENT
            and self._peek(1).type == TT.LPAREN
        ):
            # peek ahead to find matching ')' then '='
            save = self._pos
            try:
                name = self._cur.value
                col = self._cur.col
                self._advance()  # name
                self._advance()  # (
                params: list[str] = []
                if self._cur.type != TT.RPAREN:
                    params.append(self._expect(TT.IDENT).value)
                    while self._match(TT.COMMA):
                        self._advance()
                        params.append(self._expect(TT.IDENT).value)
                self._expect(TT.RPAREN)
                if self._match(TT.EQUALS):
                    self._advance()
                    body = self._parse_expr()
                    return AssignNode(name, params, body, col)
            except ParseError:
                pass
            # not a function def – backtrack
            self._pos = save

        # variable assignment: IDENT '='
        if self._cur.type == TT.IDENT and self._peek(1).type == TT.EQUALS:
            name = self._cur.value
            col = self._cur.col
            self._advance()
            self._advance()
            body = self._parse_expr()
            return AssignNode(name, [], body, col)

        return self._parse_expr()

    def _parse_expr(self) -> ASTNode:
        node = self._parse_term()
        while self._match(TT.PLUS, TT.MINUS):
            op = self._advance().value
            right = self._parse_term()
            node = BinaryNode(op, node, right)
        return node

    def _parse_term(self) -> ASTNode:
        node = self._parse_unary()
        while self._match(TT.STAR, TT.SLASH, TT.PERCENT):
            op = self._advance().value
            right = self._parse_unary()
            node = BinaryNode(op, node, right)
        return node

    def _parse_unary(self) -> ASTNode:
        if self._match(TT.MINUS, TT.PLUS):
            op = self._advance().value
            operand = self._parse_unary()
            return UnaryNode(op, operand)
        return self._parse_power()

    def _parse_power(self) -> ASTNode:
        base = self._parse_primary()
        if self._match(TT.STARSTAR):
            self._advance()
            exp = self._parse_unary()  # right-associative
            return BinaryNode("**", base, exp)
        return base

    def _parse_primary(self) -> ASTNode:
        tok = self._cur

        # Number literal
        if tok.type == TT.NUMBER:
            self._advance()
            node: ASTNode = NumberNode(float(tok.value), tok.col)
            # implicit multiplication: 2x or 2(...)
            node = self._implicit_mul(node)
            return node

        # Identifier or function call
        if tok.type == TT.IDENT:
            self._advance()
            if self._match(TT.LPAREN):
                self._advance()
                args: list[ASTNode] = []
                if not self._match(TT.RPAREN):
                    args.append(self._parse_expr())
                    while self._match(TT.COMMA):
                        self._advance()
                        args.append(self._parse_expr())
                self._expect(TT.RPAREN)
                node = CallNode(tok.value, args, tok.col)
            else:
                node = IdentNode(tok.value, tok.col)
            node = self._implicit_mul(node)
            return node

        # Parenthesised expression
        if tok.type == TT.LPAREN:
            self._advance()
            node = self._parse_expr()
            self._expect(TT.RPAREN)
            node = self._implicit_mul(node)
            return node

        # Matrix literal [[...],[...]]
        if tok.type == TT.LBRACKET:
            return self._parse_matrix()

        raise ParseError(f"Unexpected token '{tok.value}'", tok.col)

    def _implicit_mul(self, left: ASTNode) -> ASTNode:
        """Handle implicit multiplication: 2x, 2(x+1), x(y+1)."""
        while True:
            if self._match(TT.IDENT):
                right = IdentNode(self._cur.value, self._cur.col)
                self._advance()
                left = BinaryNode("*", left, right)
            elif self._match(TT.LPAREN):
                self._advance()
                inner = self._parse_expr()
                self._expect(TT.RPAREN)
                left = BinaryNode("*", left, inner)
            else:
                break
        return left

    def _parse_matrix(self) -> ASTNode:
        col = self._cur.col
        self._expect(TT.LBRACKET)
        rows: list[list[ASTNode]] = []
        while not self._match(TT.RBRACKET):
            self._expect(TT.LBRACKET)
            row: list[ASTNode] = [self._parse_expr()]
            while self._match(TT.COMMA):
                self._advance()
                row.append(self._parse_expr())
            self._expect(TT.RBRACKET)
            rows.append(row)
            if self._match(TT.COMMA):
                self._advance()
        self._expect(TT.RBRACKET)
        return MatrixNode(rows, col)

    # ------------------------------------------------------------------ public
    def parse(self) -> ASTNode:
        node = self.parse_stmt()
        if self._cur.type != TT.EOF:
            raise ParseError(
                f"Unexpected token '{self._cur.value}' after expression",
                self._cur.col,
            )
        return node


def parse(text: str) -> ASTNode:
    """Parse *text* and return an AST node."""
    tokens = tokenize(text)
    return Parser(tokens).parse()
