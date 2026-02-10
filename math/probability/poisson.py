bj#!/usr/bin/env python3
""" This module will define a class named Poisson """


class Poisson():
    """ This class will demonstrate a poisson distribution """
    def __init__(self, data=None, lambtha=1.):
        """ Function for initializing class """
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = lambtha
        elif not isinstance(data, list):
            raise TypeError('data must be a list')
        elif len(data) < 2:
            raise ValueError('data must contain multiple values')
        else:
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """ Calculates the value of PMF for given num of success """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        f = 1
        for i in range(2, k+1):
            f = f * i
        return (self.lambtha ** k) * (2.7182818285 ** (-self.lambtha)) / f

    def cdf(self, k):
        """ CDF for a given number of successes """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        cdf_val = 0
        for i in range(0, k+1):
            cdf_val += self.pmf(i)
        return cdf_val
