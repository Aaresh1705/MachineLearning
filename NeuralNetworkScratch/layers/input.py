# my_library/layers/input.py

import numpy as np
from .base import Layer


class Input(Layer):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.dim = dim

    def forward(self, inputs):
        return np.array(inputs)
