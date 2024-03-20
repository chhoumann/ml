## Resources

This folder follows: [The spelled-out intro to neural networks and backpropagation: building micrograd by Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0)

## Notes

We're building a small automatic gradient (autograd) engine.
In practice, this means implementing backpropagation with reverse-mode automatic differentiation (reverse-mode auto-diff).
Backpropagation is a way to compute gradients of expressions through a recursive application of the chain rule.
Efficiently evaluate the gradient of a loss function w.r.t to the parameters (weights) of a model (neural net).
This allows us to iteratively tune the weights of the NN to minimize the loss function, thereby improving the model's accuracy.

- Build out mathematical expression as a computation graph (DAG)
- Evaluate the graph forward to get the value of the expression
- Evaluate the graph backward to get the gradients of the expression w.r.t to its inputs
    - Start with the gradient of the loss function w.r.t to the output of the expression
    - Apply the chain rule (from Calculus) to recursively compute the gradients of the loss function w.r.t to the inputs of the expression.
    - So we get the derivative of the loss function w.r.t to the inputs of the expression.


This is all you need to build a simple autograd engine - which is all you need to train a neural network.
But this is not the most efficient way to do it.


```python
from micrograd.engine import Value

# create some values
a = Value(-4.0) # create a value object
b = Value(2.0) 

# define a simple computation
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
# forward pass: compute values
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass

# backward pass: compute gradients
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```