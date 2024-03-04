import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def step_function(x: float):
    """
    Step function for binary classification.
    y_hat = 1 if x > 0 else 0
    """
    return 1 if x > 0 else 0


class Perceptron:
    def __init__(self, n_inputs: int, learning_rate=0.1):
        self.weights = np.random.rand(n_inputs + 1)  # random initialization is fine for this example
        self.learning_rate = learning_rate

    def predict(self, x: NDArray):
        """
        Calculate the weighted sum of inputs and weights and apply the step function to get the predicted output.
        z = Wx + b
        where W is the weights (matrix), x is the input (vec), and b is the bias (vec).
        """
        weighted_sum = np.dot(x, self.weights[1:]) + self.weights[0]
        return step_function(weighted_sum)

    def train(self, x: NDArray, y: NDArray):
        """
        1. Initialize weights with random values
        2. For each input, calculate the weighted sum of inputs and weights
        3. Apply the step function to the weighted sum to get the predicted output
        4. Calculate the error (y - y_hat)
        5. Update the weights using the formula:
        W^{next step} = W + η(y - y_hat)x
        6. Update the bias using the formula:
        b^{next step} = b + η(y - y_hat)
        7. Repeat until the error is minimized
        """
        y_hat = self.predict(x)
        error = y - y_hat
        # update weights
        self.weights[1:] += self.learning_rate * error * x
        # update bias
        self.weights[0] += self.learning_rate * error


def generate_and_gate_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    return X, y


def generate_or_gate_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    return X, y


def generate_nand_gate_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1, 1, 1, 0])
    return X, y


def generate_nor_gate_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1, 0, 0, 0])
    return X, y


def generate_xor_gate_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    return X, y


def run(X: NDArray, y: NDArray, epochs=10):
    logger.debug("Training Perceptron with X: %s and y: %s", X, y)
    perceptron = Perceptron(n_inputs=X.shape[1])
    logger.debug("Perceptron initialized\n\n")

    for i in range(epochs):
        logger.debug("EPOCH: %s", i)
        for inputs, target in zip(X, y):
            perceptron.train(inputs, target)

        logger.debug("Weights after epoch %s: %s\n\n", i, perceptron.weights)

    y_pred = np.array([perceptron.predict(inputs) for inputs in X])
    logger.info("Accuracy: %s", np.mean(y_pred == y))
    logger.info("Done")


if __name__ == "__main__":
    for gate in [
        generate_and_gate_data,
        generate_or_gate_data,
        generate_nand_gate_data,
        generate_nor_gate_data,
        generate_xor_gate_data,
    ]:
        logger.info("Running perceptron for %s", gate.__name__)
        X, y = gate()
        run(X, y, 20)
        logger.info("\n\n")

    # Generate binary classification dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Running perceptron for binary classification")
    run(X_train, y_train, 20)
