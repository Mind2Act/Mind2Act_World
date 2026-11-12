"""Exact four-operation parser; only integer operands and explicit operators."""
from fractions import Fraction


def calculate(tokens):
    tokens=[{'×':'*','÷':'/','−':'-'}.get(t,t) if isinstance(t,str) else t for t in tokens]
    cursor=0

    def atom():
        nonlocal cursor
        if cursor>=len(tokens):
            raise ValueError('Missing operand')
        token=tokens[cursor]
        cursor+=1
        if type(token) is int:
            return Fraction(token)
        if token=='(':
            value=expression()
            if cursor>=len(tokens) or tokens[cursor]!=')':
                raise ValueError('Unclosed parenthesis')
            cursor+=1
            return value
        raise ValueError(f'Invalid operand: {token!r}')

    def term():
        nonlocal cursor
        value=atom()
        while cursor<len(tokens) and tokens[cursor] in ('*','/'):
            operation=tokens[cursor]
            cursor+=1
            other=atom()
            value=value*other if operation=='*' else value/other
        return value

    def expression():
        nonlocal cursor
        value=term()
        while cursor<len(tokens) and tokens[cursor] in ('+','-'):
            operation=tokens[cursor]
            cursor+=1
            other=term()
            value=value+other if operation=='+' else value-other
        return value

    result=expression()
    if cursor!=len(tokens):
        raise ValueError('Unexpected trailing tokens')
    return result


def answer_digits(tokens):
    value=calculate(tokens)
    if value.denominator!=1 or value<0:
        raise ValueError('The task requires a non-negative integer answer')
    digits=[int(d) for d in str(value.numerator)]
    if not all(1<=digit<=9 for digit in digits):
        raise ValueError('Answer outside the nine-key numeric keypad')
    return digits
