#!/usr/bin/env python3
""" Write a function for calculating sensitivity of confusion matrix """
import numpy as np


def sensitivity(confusion):
    """ Function for calculating sensitivity """
    classes = confusion.shape[0]
    arr = np.zeros(classes)
    for i in range(classes):
        tp_fn = np.sum(confusion[i])
        tp = confusion[i][i]
        arr[i] = tp / tp_fn
    return arr
