#!/usr/bin/env python3
""" This module will define a function which will calculate F1 """
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ Function for calculating F1-score """
    return 2 / (1 / sensitivity(confusion) + 1 / precision(confusion))
