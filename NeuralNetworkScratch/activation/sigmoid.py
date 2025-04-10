# NeuralNetworkScratch/activation/sigmoid.py

import numpy as np
from .base import ActivationFunction


class Sigmoid(ActivationFunction):
    def forward(self, input):
        return 1 / (1 + np.exp(-input))

    def backward(self, input):
        z = self.forward(input)
        return z * (1 - z)
