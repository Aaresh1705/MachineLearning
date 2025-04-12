import numpy as np
from .base import ActivationFunction

class Sin(ActivationFunction):
    def forward(self, inputs):
        """
        Compute the element-wise sine of 'input'.
        """
        self.inputs = inputs
        self.output = np.sin(inputs)
        return self.output

    def backward(self, d_values):
        """
        The derivative of sin(x) is cos(x).
        """
        self.d_inputs = d_values * np.cos(self.inputs)  # ∂sin(x)/∂x = cos(x)
        return self.d_inputs

    def latex(self):
        return r'$\sin(x)$'
