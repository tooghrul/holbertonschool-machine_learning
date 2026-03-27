#!/usr/bin/env python3
""" This module will define a function for creating confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """ This function will create a confusion matrix """
    m, classes = labels.shape
    confusion_matrix = np.zeros((classes, classes))
    for sample in range(m):
        true_class = np.argmax(labels[sample])
        pred_class = np.argmax(logits[sample])
        confusion_matrix[true_class][pred_class] += 1
    return confusion_matrix
