#!/usr/bin/env python3
"""The list of coefficients of polynomials"""
def poly_derivative(poly):
    """Implementation"""
    new_poly [0 for i in range(len(poly)-1)]
    for i in range(len(new_poly)-1):
        new_poly[i] = (i+1)*poly[i+1]
    
    if len(poly) == 1:
        return [0]

    return new_poly
