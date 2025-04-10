# my_library/layers/dense.py

from typing import Type
import numpy as np
from sympy.vector import DyadicZero

from .base import Layer
from ..activation.base import ActivationFunction
from ..activation.linear import Linear


class Dense(Layer):
    def __init__(self,
                 dim: int,
                 activation: Type[ActivationFunction] = Linear
                 ):
        super().__init__(dim, activation)

    def initializeWeights(self, dim_before):
        self.weights = np.random.random((dim_before, self.dim))
        self.biases = np.random.random(self.dim)

    def forward(self, inputs):
        self.inputs = inputs

        self.output = np.dot(inputs, self.weights) + self.biases

        return self.output

    def backward(self, d_values):
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0)

        self.d_inputs = np.dot(d_values, self.weights.T)

        return self.d_inputs
