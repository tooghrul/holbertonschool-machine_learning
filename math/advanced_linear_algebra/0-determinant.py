#!/usr/bin/env python3
""" This module will define a function for calculating determinant """


def determinant(matrix):
    """ Function for calculating determinant """
    def new_matrix(matrix, col):
        """ return new_matrix for laplace expansion """
        return [
            row[:col] + row[col+1:]
            for row in matrix[1:]
        ]
    if matrix == [[]]:
        return 1
    if not isinstance(matrix, list) or not all(isinstance(row, list)
       for row in matrix):
        raise TypeError('matrix must be a list of lists')
    length = len(matrix)

    for row in matrix:
        if len(row) != length:
            raise ValueError('matrix must be a square matrix')

    if length == 1:
        return matrix[0][0]

    if length == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    D = 0
    for j in range(length):
        D += matrix[0][j] * (-1) ** j * determinant(new_matrix(matrix, j))
    return D
