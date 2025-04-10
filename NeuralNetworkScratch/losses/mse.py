# NeuralNetworkScratch/losses/mse.py

import numpy as np
from .base import Loss


class MSE(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        sample_loss = np.mean((y_true - y_pred)**2, axis=1)

        return sample_loss

    def backward(self, d_values, y_true):
        samples = len(d_values)
        outputs = len(d_values[0])

        self.d_inputs = -2 * (y_true - d_values) / outputs

        self.d_inputs = self.d_inputs / samples

        return self.d_inputs

    def __bool__(self):
        return True
