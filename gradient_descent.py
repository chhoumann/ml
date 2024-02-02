import numpy as np


def batch_gradient_descent(X, y, learning_rate, num_iterations):
    """
    Perform batch gradient descent to optimize the parameters theta for linear regression.

    Parameters:
    X (ndarray): The input features matrix of shape (m, n), where m is the number of samples and n is the number of features.
    y (ndarray): The target values of shape (m,).
    learning_rate (float): The learning rate (eta) for gradient descent.
    num_iterations (int): The number of iterations to perform. Epochs.

    Returns:
    ndarray: The optimized parameters theta of shape (n,).
    """
    m = len(y)
    theta = np.zeros(X.shape[1])

    for _ in range(num_iterations):
        predictions = np.dot(X, theta)
        errors = predictions - y
        gradient = np.dot(X.T, errors) / m
        theta -= learning_rate * gradient

    return theta
