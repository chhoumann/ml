import numpy as np
import numpy.linalg as LA


def dot_product(v1, v2):
    return sum(v1i * v2i for v1i, v2i in zip(v1, v2))


def euclid_norm(v1):
    """
    Finds the Euclidean norm of the vector
    """
    res = sum(x**2 for x in v1)
    return res**0.5


def vector_angle(u, v):
    """
    Calculates angle between two vectors
    """
    cos_theta = u.dot(v) / LA.norm(u) / LA.norm(v)
    # cos_theta may be outside the [-1, 1] interval due to small floating
    # point errors
    # This could make arccos fail. So we clip the value within the range.
    return np.arccos(cos_theta.clip(-1, 1))


def project_vec(u, v):
    return (u.dot(v) / euclid_norm(u) ** 2) * u


def least_squares_fit(x, y):
    n = len(x)

    # calculate sums
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_squared = sum([xi**2 for xi in x])
    sum_xy = sum([xi * yi for xi, yi in zip(x, y)])

    # calculate slope (a) and y-intercept (b) using the formulas
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
    b = (sum_y - a * sum_x) / n

    return a, b


x = [1, 2, 3, 4, 5]
y = [2.1, 3.9, 6.1, 8.0, 10.2]
print(least_squares_fit(x, y))
