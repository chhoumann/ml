def matching_types(x, y):
    if not isinstance(y, x.__class__):
        raise TypeError(
            f"unsupported operand type(s) for +: '{x.__class__}' and '{type(y)}'"
        )


class Matrix:
    def __init__(self, inputs, dtype=float):
        self.matrix = inputs  # could cast to dtype here to enforce usage of it
        self.dtype = dtype
        self.shape = (
            len(inputs),
            len(inputs[0]),
        )

    def __get_transpose__(self):
        return Matrix(list(map(list, zip(*self.matrix))))

    T = property(fget=__get_transpose__)

    def determinant(self):
        if self.shape[0] != self.shape[1]:
            raise ValueError(
                "Determinant can only be calculated for square matrices"
            )
        return self.__calculate_determinant__(self.matrix)

    def __calculate_determinant__(self, matrix):
        if len(matrix) == 1:
            return matrix[0][0]
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        det = 0
        for i in range(len(matrix)):
            minor = [row[:i] + row[i + 1:] for row in matrix[1:]]
            cofactor = matrix[0][i] * ((-1) ** i)
            det += cofactor * self.__calculate_determinant__(minor)
        return det

    def inverse(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, can't find inverse")
        return self.__calculate_inverse__(self.matrix, det)

    def __calculate_inverse__(self, matrix, det):
        n = len(matrix)
        adjoint = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                minor = [
                    row[:j] + row[j + 1:]
                    for row in (matrix[:i] + matrix[i + 1:])
                ]
                cofactor = ((-1) ** (i + j)) * self.__calculate_determinant__(
                    minor
                )
                adjoint[j][i] = cofactor / det
        return Matrix(adjoint)

    def __add__(self, other):
        return Matrix(
            self.__elementwise_binary_op__(lambda x, y: x + y, other)
        )

    def __sub__(self, other):
        return Matrix(
            self.__elementwise_binary_op__(lambda x, y: x - y, other)
        )

    def __eq__(self, other):
        res = self.__elementwise_binary_op__(lambda x, y: x == y, other)
        return all(all(row) for row in res)

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            # Scalar multiplication
            return Matrix([[x * other for x in row] for row in self.matrix])
        matching_types(self, other)
        assert self.shape[1] == other.shape[0], "Shape mismatch"
        return self.__matrix_multiply__(other)

    def __matrix_multiply__(self, other):
        rows, cols = self.shape[0], other.shape[1]
        m1, m2 = self.matrix, other.matrix
        res = []
        for row_i in range(rows):
            row_res = []
            v1 = m1[row_i]
            for col_i in range(cols):
                v2 = [m2[k][col_i] for k in range(len(v1))]
                row_res.append(sum(x * y for x, y in zip(v1, v2)))
            res.append(row_res)

        return Matrix(res)

    def __elementwise_binary_op__(self, op, other):
        matching_types(self, other)
        assert (
            self.shape == other.shape
        ), f"Shape mismatch: {self.shape} vs {other.shape}"
        return [
            [op(x1, x2) for x1, x2 in zip(v1, v2)]
            for v1, v2 in zip(self.matrix, other.matrix)
        ]

    def __str__(self):
        return "\n".join(
            [" ".join(map(lambda x: str(x), vec)) for vec in self.matrix]
        )


def main():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    B = Matrix([[10, 20, 30], [40, 50, 60]])

    print(f"{(A + B)=}")
    print()
    print(A - B)
    print()
    print(A == B, A == A)
    print()

    C = Matrix([[1, 2], [3, 4], [5, 6]])
    print(A * C)

    D = Matrix([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
    print(f"Inverse of D:\n{D.inverse()}")
    print()
    print(D.determinant())
    print()
    print(D.T)


if __name__ == "__main__":
    raise SystemExit(main())
