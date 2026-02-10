#!/usr/bin/env python3
"""task 8"""


def mat_mul(mat1, mat2):
    def matrix_shape(matrix):
    """Matrix shape"""
    size = []
    if not isinstance(matrix, list):
        return size
    else:
        size.append(len(matrix))
        matrix = matrix[0]
        return size + matrix_shape(matrix)

    mat1_shape = matrix_shape(mat1)
    mat2_shape = matrix_shape(mat2)

    if mat1_shape[1] != mat2_shape[0]:
        return None

    newMat_shape = [mat1_shape[0], mat2_shape[1]]
    newMat = [[ for i in range(newMat_shape[1])] for j in range(newMat_shape[0])] #duzelt
