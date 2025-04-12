# mycelium/optimizers/sdg.py

from .base import Optimizer


class SGD(Optimizer):
    def __init__(self, learning_rate=1.0):
        super().__init__(learning_rate)

    def update(self, layer):
        layer.weights -= self.learning_rate * layer.d_weights
        layer.biases -= self.learning_rate * layer.d_biases

        """
        grad_weight, grad_bias, new_grad = layer.backward(grad, a)
        layer.weights -= self.lr * grad_weight.reshape(layer.weights.shape)
        layer.biases  -= self.lr * grad_bias

        return new_grad
        """

    def __bool__(self):
        return True
