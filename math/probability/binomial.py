#!/usr/bin/env python3
""" This module will define a class Binomial """


class Binomial():
    """ Class for representing binomial distribution """
    def __init__(self, data=None, n=1, p=0.5):
        """ Initialize class """
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = n
            self.p = p
        elif not isinstance(data, list):
            raise TypeError('data must be a list')
        elif len(data) < 2:
            raise ValueError('data must contain multiple values')
        else:
            mean = sum(data) / len(data)
            variance = 0
            for trail in data:
                variance += (trail - mean) ** 2
            variance = variance / len(data)
            p = 1 - variance / mean
            n = round(mean / p)
            p = mean / n
            self.p = p
            self.n = n

    def pmf(self, k):
        """ Calculate pmf for given value k """
        def f(n):
            """ Helper factorial func """
            h = 1
            for i in range(2, n+1):
                h = h * i
            return h
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        coef = f(self.n) / f(k) / f(self.n - k)
        return coef * self.p ** k * (1 - self.p) ** (self.n - k)

    def cdf(self, k):
        """ Calculate cdf for given value k """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        if k >= self.n:
            return 1
        cdf_val = 0
        for i in range(0, k+1):
            cdf_val += self.pmf(i)
        return cdf_val
