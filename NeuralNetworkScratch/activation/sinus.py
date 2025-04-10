import numpy as np
from .base import ActivationFunction

class Sin(ActivationFunction):
    def forward(self, input):
        """
        Compute the element-wise sine of 'input'.
        """
        return np.sin(input)

    def backward(self, input):
        """
        The derivative of sin(x) is cos(x).
        """
        return np.cos(input)
