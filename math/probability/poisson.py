#!/usr/bin/env python3
class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        # Case 1: data is None → use lambtha
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
            return

        # Case 2: data is given
        if not isinstance(data, list):
            raise TypeError("data must be a list")

        if len(data) < 2:
            raise ValueError("data must contain multiple values")

        # Estimate lambtha from data (mean)
        self.lambtha = float(sum(data) / len(data))

