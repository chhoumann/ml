import numpy as np
import matplotlib.pyplot as plt


def mse_cost(x, y, theta_0, theta_1):
    """
    Calculates the mean squared error (MSE) cost and the error for linear regression.

    Parameters:
    x (array-like): The input feature values.
    y (array-like): The target values.
    theta_0 (float): The intercept term.
    theta_1 (float): The slope term.

    Returns:
    cost (float): The mean squared error cost.
    error (array-like): The difference between the predicted and actual values.
    """
    m = len(y)
    predicted = theta_0 + theta_1 * x

    error = predicted - y
    cost = (1 / (2 * m)) * np.sum(error**2)

    return cost, error


def result(x, theta_0, theta_1):
    """
    Calculate the result of a linear regression model.

    Parameters:
    x (float): The input value.
    theta_0 (float): The intercept of the linear regression model.
    theta_1 (float): The slope of the linear regression model.

    Returns:
    float: The result of the linear regression model.
    """
    return theta_0 + theta_1 * x


def gradient_descent(
    x, y, theta_0, theta_1, alpha, num_iters, cost_fn=mse_cost
):
    """
    Perform gradient descent to optimize the parameters theta_0 and theta_1
    for linear regression.

    Parameters:
    - x: Input features (array-like)
    - y: Target values (array-like)
    - theta_0: Initial value for parameter theta_0 (float)
    - theta_1: Initial value for parameter theta_1 (float)
    - alpha: Learning rate (float)
    - num_iters: Number of iterations for gradient descent (int)
    - cost_fn: Cost function to compute the cost and error (function)

    Returns:
    - theta_0: Optimized value for parameter theta_0 (float)
    - theta_1: Optimized value for parameter theta_1 (float)
    - cost_history: Array of cost values at each iteration (array-like)
    """
    m = len(y)
    cost_history = np.zeros(num_iters)

    for i in range(num_iters):
        cost, error = cost_fn(x, y, theta_0, theta_1)

        # compute the gradients
        grad_theta_0 = (1 / m) * sum(error)
        grad_theta_1 = (1 / m) * sum(error * x)

        # update parameters
        theta_0 = theta_0 - alpha * grad_theta_0
        theta_1 = theta_1 - alpha * grad_theta_1

        cost_history[i] = cost

    return theta_0, theta_1, cost_history


data = [
    [1000, 200_000],
    [2000, 250_000],
    [4000, 300_000],
]


def main():
    # separate into two lits, one for sqft and one for prices
    x = np.array([i[0] for i in data])
    y = np.array([i[1] for i in data])

    # save original mean and std
    x_mean = np.mean(x)
    x_std = np.std(x)
    y_mean = np.mean(y)
    y_std = np.std(y)

    # normalize the data
    x = (x - x_mean) / x_std
    y = (y - y_mean) / y_std

    theta_0 = 0
    theta_1 = 0

    alpha = 0.01
    num_iters = 1000

    theta_0, theta_1, cost_history = gradient_descent(
        x, y, theta_0, theta_1, alpha, num_iters, mse_cost
    )

    print("theta_0 (normalized):", theta_0)
    print("theta_1 (normalized):", theta_1)
    print(
        "Final cost (normalized):", cost_history[-1]
    )  # The cost at the final iteration

    # un-normalize thetas
    theta_0_unnorm = (
        y_std * theta_0 + y_mean - theta_1 * y_std * x_mean / x_std
    )
    theta_1_unnorm = theta_1 * y_std / x_std

    print("theta_0 (unnormalized):", theta_0_unnorm)
    print("theta_1 (unnormalized):", theta_1_unnorm)

    # un-normalize data for plotting
    x_unnorm = x * x_std + x_mean
    y_unnorm = y * y_std + y_mean

    x_range_unnorm = np.linspace(min(x_unnorm), max(x_unnorm), num=100)
    x_range_norm = (x_range_unnorm - x_mean) / x_std  # normalize x_range

    y_pred_norm = (
        theta_0 + theta_1 * x_range_norm
    )  # generate predictions in normalized space
    y_pred_unnorm = y_pred_norm * y_std + y_mean  # un-normalize predictions

    plt.scatter(x_unnorm, y_unnorm)  # Original data
    plt.plot(x_range_unnorm, y_pred_unnorm, color="red")  # Regression line

    plt.xlabel("Square Footage")
    plt.ylabel("Price")
    plt.savefig("plot.png")  # Save the plot to a file


if __name__ == "__main__":
    main()
