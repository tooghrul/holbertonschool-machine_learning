#!/usr/bin/env python3

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
        """Calculates the PMF for a given number of successes"""
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        # factorial(k)
        fact = 1
        for i in range(1, k + 1):
            fact *= i

        # Poisson PMF: (λ^k * e^(-λ)) / k!
        e = 2.718281828459045
        return (self.lambtha ** k * (e ** (-self.lambtha))) / fact
