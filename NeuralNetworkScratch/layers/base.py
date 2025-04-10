# NeuralNetworkScratch/layers/base.py

from typing import Type
from ..activation.base import ActivationFunction


class Layer:
    def __init__(self,
                 dim: int,
                 activation: Type[ActivationFunction] = ActivationFunction
                 ):
        self.dim = dim
        self.activation = activation()
        self.weights = []
        self.d_weights = []
        self.biases = []
        self.d_biases = []
        self.inputs = None
        self.d_inputs = None
        self.output = None

    def call(self, inputs):
        raise NotImplementedError("Subclasses must implement the call method.")

    def __call__(self, inputs):
        # This allows the layer instance to be callable like a function.
        return self.call(inputs)

    def forward(self, inputs):
        raise NotImplementedError("Subclasses must implement the call method.")

    def initializeWeights(self, shape_before):
        raise NotImplementedError("Subclasses must implement the initializeWeights method.")

    def backward(self, d_values):
        raise NotImplementedError("Subclasses must implement the call method.")
