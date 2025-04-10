# NeuralNetworkScratch/activation/base.py

class ActivationFunction:
    """
    hello
    """
    def __init__(self):
        self.d_inputs = None
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        raise NotImplementedError("Subclasses must implement the call method.")

    def backward(self, d_values):
        raise NotImplementedError("Subclasses must implement the call method.")

    def __call__(self, inputs):
        """
        Enables the activation function to be called as a function.

        This method allows an instance of the activation function to be used as a callable,
        forwarding the inputs to the forward() method. For example:

        >>> activation = SomeActivationFunctionSubclass()
        >>> output = activation(input_data)

        :param inputs: array-like
            The input data to be processed by the activation function.
        :return: array-like
            The output after applying the activation function.
        """
        return self.forward(inputs)
