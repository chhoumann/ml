from typing import List

from grad.engine import Value
from grad.nn import MLP

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired targets


def gradient_descent(model: MLP, n_epochs: int, learning_rate: float):
    for k in range(n_epochs):
        # Forward pass
        y_pred = [model(x) for x in xs]
        loss: Value = sum([(yout - ygt) ** 2 for ygt, yout in zip(ys, y_pred)])  # type: ignore

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Update. This is very simple - in practice we would use an optimizer
        # like Adam or SGD with momentum.
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 100 == 0:
            print(f"Epoch {k} Loss: {loss.data:.4f}")
        elif k == n_epochs - 1:
            print(f"Epoch {k} Loss: {loss.data:.4f}")


def main():
    mlp = MLP(3, [4, 4, 1])
    gradient_descent(mlp, n_epochs=1000, learning_rate=0.1)

    y_pred: List[Value] = [mlp(x) for x in xs]  # type: ignore

    print("Final predictions:", [y_pred.data for y_pred in y_pred])


if __name__ == "__main__":
    main()
