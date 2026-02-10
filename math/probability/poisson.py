#!/usr/bin/env python3


class Poisson:
    """
    Represents a Poisson distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Poisson distribution

        data: list of observed data points
        lambtha: expected number of occurrences
        """
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

        k: number of successes
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        # Calculate factorial of k
        fact = 1
        for i in range(1, k + 1):
            fact *= i

        # Euler's number approximation
        e = 2.718281828459045

        return (self.lambtha ** k * (e ** (-self.lambtha))) / fact

    def cdf(self, k):
        """
        Calculates the CDF for a given number of successes

        k: number of successes
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        cdf_value = 0
        for i in range(0, k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
