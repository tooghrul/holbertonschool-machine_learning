#!/usr/bin/env python3
"""17-integrate"""


def poly_integral(poly, C=0):
    """The implementation"""
    if not isinstance(poly, list) or not isinstance(C, int):
        return None
    if poly == []:
        return None

    result = [C]

    for i, coeff in enumerate(poly):
        if not isinstance(coeff, (int, float)):
            return None
        value = coeff / (i + 1)
        if value.is_integer():
            value = int(value)
        result.append(value)

    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result
