#!/usr/bin/env python3

def summation_i_squared(n):
    if not isinstance(n, int) or n < 1:
        return None

    return (n*(n+1)*(2*n+1)) / 6
