#!/usr/bin/env python3
"""task 5"""


def add_matrices2D(mat1, mat2):
    """Task 5"""
    def matrix_shape(matrix):
        size = []
        if not isinstance(matrix, list):
            return size
        else:
            size.append(len(matrix))
            matrix = matrix[0]
            return size + matrix_shape(matrix)
    if (matrix_shape(mat1) != matrix_shape(mat2)) or ((len(mat1) == 0) and (len(mat2) == 0)):
        return None

    new_matrix = [[mat1[i][j]+mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
    return new_matrix
