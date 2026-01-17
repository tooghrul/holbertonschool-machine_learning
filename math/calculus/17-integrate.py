#!/usr/bin/env python3
"""17-integrate"""

def poly_integral(poly, C=0):
    if not (isinstance(poly, list) or isinstance(C, int)):
        return None
    
    new_poly = [0 for i in range(len(poly) + 1)]
    for i in range(1, len(new_poly)):
        new_poly[i] = poly[i-1]/(i+1)
    return new_poly
