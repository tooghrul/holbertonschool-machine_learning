#!/usr/bin/env python3

def matrix_shape(matrix):
    size = []
    if not isinstance(matrix, list):
        return size
    else:
        size.append(len(matrix))
        matrix = matrix[0]
        return size + matrix_shape(matrix)
