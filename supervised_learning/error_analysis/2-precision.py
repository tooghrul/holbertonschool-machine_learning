#!/usr/bin/env python3
""" Calculate precision of each class in confusion matrix """
import numpy as np


def precision(confusion):
    """ Calculate precision """
    classes = len(confusion)
    arr = np.zeros(classes)
    for i in range(classes):
        tp_fp = np.sum(confusion[:, i])
        tp = confusion[i][i]
        arr[i] = tp / tp_fp
    return arr
