#!/usr/bin/env python3

import math


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
            return

        if not isinstance(data, list):
            raise TypeError("data must be a list")

        if len(data) < 2:
            raise ValueError("data must contain multiple values")

        self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the PMF for a given number of successes
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        # PMF formula: (λ^k * e^(-λ)) / k!
        return (self.lambtha ** k * math.exp(-self.lambtha)) / math.factorial(k)
