# NeuralNetworkScratch/activation/relu.py

import numpy as np
from .base import ActivationFunction


class Relu(ActivationFunction):
    """
    Implementation of the Rectified Linear Unit (ReLU) activation function.

    The ReLU activation function is defined as:
        f(x) = max(0, x)
    This function outputs 0 for all negative input values and returns the input itself for positive values.
    It is widely used in neural networks to introduce non-linearity and to help alleviate the vanishing gradient problem.
    """

    def __init__(self):
        """
        Initializes the ReLU activation function.

        Calls the initializer of the parent ActivationFunction class.
        """
        super().__init__()

    def forward(self, inputs):
        """
        Performs the forward pass of the ReLU activation function.

        Applies the element-wise ReLU function on the input data by replacing negative values with 0.

        :param inputs: np.ndarray
            The input array to the activation function. Can be of any shape.
        :return: np.ndarray
            The output array after applying ReLU activation, where each element is computed as max(0, x).

        :example:
            >>> relu = Relu()
            >>> x = np.array([-1, 0, 3, -5])
            >>> relu.forward(x)
            array([0, 0, 3, 0])
        """
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

        return self.output

    def backward(self, inputs):
        self.d_inputs = inputs.copy()

        self.d_inputs[self.d_inputs <= 0] = 0

        return self.d_inputs


if __name__ == "__main__":
    relu = Relu()
    x = np.array([-1, 0, 3, -5])
    y_forward = relu.forward(x)
    print(y_forward)

    y_call = relu(x)
    print(y_call)
