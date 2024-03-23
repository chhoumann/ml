import random
from typing import List

from grad.engine import Value


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


class Neuron(Module):
    def __init__(self, nin: int):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w.x+b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin: int, nouts: List[int]):
        sz = [
            nin
        ] + nouts  # e.g. when we have nin=3 and nouts=[4, 4, 1], sz=[3, 4, 4, 1]
        self.layers = [
            Layer(sz[i], sz[i + 1]) for i in range(len(nouts))
        ]  # create a layer for each pair of sizes
        # e.g. create a layer for 3->4, 4->4, 4->1

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
