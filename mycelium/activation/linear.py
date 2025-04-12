# mycelium/activation/relu.py (renamed or replaced to represent linear)

import numpy as np
from .base import ActivationFunction


class Linear(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.inputs = input
        self.output = input

        return self.output

    def backward(self, d_values):
        # Derivative of f(x) = x is 1 for all x
        self.d_inputs = d_values.copy()

        return self.d_inputs

    def latex(self):
        return r"$f(x) = x$"