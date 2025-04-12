# mycelium/losses/mse.py

import numpy as np
from .base import Loss


class MSE(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        sample_loss = np.mean((y_true - y_pred)**2)
        return sample_loss

    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        outputs = len(y_pred[0])

        self.d_inputs = -2 * (y_true - y_pred) / outputs

        self.d_inputs = self.d_inputs / samples

        return self.d_inputs

    def __bool__(self):
        return True
