#!/usr/bin/env python3

import numpy as np

def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects.
    """
    # 1. Input Validation for n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # 2. Input Validation for x
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    # 3. Validation for relationship between x and n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # 4. Input Validation for P (Type and Dimensions)
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # 5. Input Validation for P (Value Range)
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # 6. Calculate the Binomial Coefficient (n Choose x)
    # We use np.math.factorial to compute n! / (x! * (n-x)!)
    fact_n = np.math.factorial(n)
    fact_x = np.math.factorial(x)
    fact_diff = np.math.factorial(n - x)
    combination = fact_n / (fact_x * fact_diff)

    # 7. Calculate Likelihood using the Binomial Probability Formula
    # L = (nCx) * (P^x) * ((1-P)^(n-x))
    L = combination * (P ** x) * ((1 - P) ** (n - x))

    return L
