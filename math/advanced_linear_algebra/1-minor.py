#!/usr/bin/env python3
""" This module will define a function named minor """


def minor(matrix):
    """ Function for calculating minor matrix of a matrix """

    def new_matrix(matrix, row_id, col_id):
        """ Minor of a matrix """
        return [
            row[:col_id] + row[col_id+1:]
            for row in (matrix[:row_id] + matrix[row_id+1:])
        ]

    def determinant(matrix):
        """ Function for calculating determinant """

        length = len(matrix)

        if length == 0:
            return 1
        if length == 1:
            return matrix[0][0]
        if length == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        D = 0
        for j in range(length):
            det = determinant(new_matrix(matrix, 0, j))
            D += matrix[0][j] * (-1) ** j * det
        return D

    if not isinstance(matrix, list) or not all(isinstance(row, list)
       for row in matrix):
        raise TypeError('matrix must be a list of lists')

    length = len(matrix)

    for row in matrix:
        if len(row) != length:
            raise ValueError('matrix must be a non-empty square matrix')

    return [
        [
            determinant(new_matrix(matrix, i, j))
            for j in range(length)
        ]
        for i in range(length)
    ]
