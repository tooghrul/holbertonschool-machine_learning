#!/usr/bin/env python3
"""linalg task 3"""


def matrix_transpose(matrix):
    """Transposing matrixes"""
    rows = len(matrix)
    col = len(matrix[0])
    new_matrix = [[matrix[j][i] for j in range(rows)] for i in range(col)]
    return new_matrix
