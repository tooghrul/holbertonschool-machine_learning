#!/usr/bin/env python3
""" Write a function for calculating specificity of confusion matrix """
import numpy as np


def specificity(confusion):
    """ Function for calculating specificity """
    classes = confusion.shape[0]
    arr = np.zeros(classes)
    for i in range(classes):
        fp = np.sum(confusion[:, i]) - confusion[i, i]
        tp = tp = confusion[i][i]
        fn = np.sum(confusion[i]) - confusion[i][i]
        N = np.sum(confusion)
        tn = N - fn - tp - fp
        tn_fp = tn + fp
        arr[i] = tn / tn_fp
    return arr
