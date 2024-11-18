import pytest
from linalg_basics import Matrix, matching_types


# Test for matrix addition
def test_matrix_addition():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    C = A + B
    assert C == Matrix([[6, 8], [10, 12]])


# Test for matrix subtraction
def test_matrix_subtraction():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    C = B - A
    assert C == Matrix([[4, 4], [4, 4]])


# Test for scalar multiplication
def test_scalar_multiplication():
    A = Matrix([[1, 2], [3, 4]])
    B = A * 2
    assert B == Matrix([[2, 4], [6, 8]])


# Test for matrix multiplication
def test_matrix_multiplication():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    C = A * B
    assert C == Matrix([[19, 22], [43, 50]])


# Test for transpose
def test_matrix_transpose():
    A = Matrix([[1, 2], [3, 4]])
    assert A.T == Matrix([[1, 3], [2, 4]])


# Test for determinant
def test_determinant():
    A = Matrix([[1, 2], [3, 4]])
    assert A.determinant() == -2


# Test for inverse
def test_inverse():
    A = Matrix([[1, 2], [3, 4]])
    assert A.inverse() == Matrix([[-2, 1], [1.5, -0.5]])


# Test for type matching utility
def test_matching_types():
    A = Matrix([[1, 2], [3, 4]])
    B = "Not a Matrix"
    with pytest.raises(TypeError):
        matching_types(A, B)
