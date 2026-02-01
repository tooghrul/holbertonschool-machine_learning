#!/usr/bin/env python3
"""Linalg task 2"""


def matrix_shape(matrix):
    """Matrix shape"""
    size = []
    if not isinstance(matrix, list):
        return size
    else:
        size.append(len(matrix))
        matrix = matrix[0]
        return size + matrix_shape(matrix)
