# my_library/layers/dense.py

from typing import Type
import numpy as np

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
        self.weights = 0.1 * np.random.rand(dim_before, self.dim)
        self.biases = np.zeros((1, self.dim))

    def forward(self, inputs):
        self.inputs = inputs

        self.output = np.dot(inputs, self.weights) + self.biases

        self.output = self.activation(self.output)

        return self.output

    def backward(self, d_values):
        d_values = self.activation.backward(d_values)

        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)

        self.d_inputs = np.dot(d_values, self.weights.T)

        return self.d_inputs
