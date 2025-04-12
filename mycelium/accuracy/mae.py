# mycelium/losses/mse.py

import numpy as np
from .base import Accuracy


class MAE(Accuracy):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        sample_loss = np.mean(np.abs(y_true - y_pred))

        return sample_loss

    def __bool__(self):
        return True
