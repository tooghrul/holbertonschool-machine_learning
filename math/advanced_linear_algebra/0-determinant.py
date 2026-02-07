#!/usr/bin/env python3
"""Determinant"""


def determinant(matrix):
    """Implementation"""
    def is_square(matrix):
        """Checking if the matrix is square"""
        row_length = [len(i) for i in matrix]
        if len(set(row_length)) != 1 or 0 in row_length:
            raise ValueError("matrix must be a square matrix")
        elif row_length[0] != len(matrix):
            raise ValueError("matrix must be a square matrix")

    def is_matrix(matrix):
        """Checking matrix type"""
        list_of_instance = [isinstance(i, list) for i in matrix]
        if False in list_of_instance or len(list_of_instance) == 0:
            raise TypeError("matrix must be a list of lists")

    def get_size(matrix):
        """Getting the size of matrix"""
        return [len(matrix), len(matrix[0])]

    def def2(matrix):
        """Calculating the determinant of 2x2 matrix"""
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    def def3(matrix):
        """Calculating the determinant of 3x3 matrix"""
        det = 0
        for i in range(3):
            minor = [[matrix[j][k] for k in range(3) if k != i] for j in range(1,3)]
            det += ((-1)**i) * matrix[0][i] * det2(minor)
        return det
    if matrix is [[]]:
        return 1

    is_matrix(matrix)
    is_square(matrix)
    size = get_size(matrix)
    if size is [1,1]:
        return matrix[0][0]

    if size is [2,2]:
        return det2(matrix)

    if size is [3,3]:
        return det3(matrix)
