def matrix_shape(matrix):
    size = []
    if not isinstance(matrix, list):
        return size
    else:
        size.append(len(matrix))
        matrix = matrix[0]
        return matrix_shape(matrix)

if __name__ == "__main__":
    matrix_shape(matrix)
