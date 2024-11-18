def det(arr: list):
    assert len(arr) == 4
    return arr[0] * arr[3] - arr[2] * arr[1]


def determinant_method(A: tuple, B: tuple):
    """
    Solves systems of equations of the form
    a_1x+b_1y=c_1
    a_2x+b_2y=c_2

    That is, two equations with two unknowns

    Input should be
    [a_1, b_1, c_1], [a_2, b_2, c_2]
    """
    assert len(a) == len(b)
    a_1, b_1, c_1 = A
    a_2, b_2, c_2 = B

    D = det([a_1, b_1, a_2, b_2])
    Dx = det([c_1, b_1, c_2, b_2])
    Dy = det([a_1, c_1, a_2, c_2])

    x = Dx // D
    y = Dy // D

    return {"x": x, "y": y}


if __name__ == "__main__":
    a, b = (-3, 1, 1), (-8, 2, -10)
    print(determinant_method(a, b))
