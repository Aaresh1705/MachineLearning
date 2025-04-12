# mycelium/models/model.py
import numpy as np
from tqdm import tqdm

from ..accuracy.base import Accuracy
from ..layers.base import Layer
from ..losses.base import Loss
from ..optimizers.base import Optimizer


class Model:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
        self.initializeWeights()

        self.optimizer = Optimizer()
        self.loss = Loss()
        self.accuracy = Accuracy()

    def initializeWeights(self):
        last_layer = self.layers[0]
        for layer in self.layers[1:]:
            layer.initializeWeights(last_layer.dim)
            last_layer = layer

    def forward(self, inputs, *, get_hidden=False):
        neurons = []

        inputs = self.layers[0].forward(inputs)
        neurons.append(inputs)
        self.optimizer.pre_update_params()
        for layer in self.layers[1:]:
            inputs = layer.forward(inputs)
            neurons.append(inputs)
        self.optimizer.post_update_params()
        if get_hidden:
            return neurons
        return neurons[-1]

    def backward(self, output, y_true):
        self.optimizer.pre_update_params()
        d_values = self.loss.backward(output, y_true)
        for layer in reversed(self.layers[1:]):
            d_values = layer.backward(d_values)
        for layer in self.layers[1:]:
            self.optimizer.update(layer)
        self.optimizer.post_update_params()

    def fit(self, train, test, epochs):
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Please set an optimizer before calling fit().")
        if self.loss is None:
            raise ValueError("Loss not set. Please set an loss before calling fit().")

        test_acc = 0
        run = tqdm(range(epochs))
        for epoch in run:
            for batch_x, batch_y in train:
                last_layer = self.forward(batch_x)

                loss = self.loss.forward(last_layer, batch_y)
                acc = self.accuracy(last_layer, batch_y)

                run.set_description(f'Loss: {loss:.4f}, Tests accuracy: {test_acc:.4f}, Lr: {self.optimizer.current_learning_rate:.4f}')

                self.backward(last_layer, batch_y)

            for batch_x, batch_y in test:
                last_layer = self.forward(batch_x)
                test_acc = self.accuracy(last_layer, batch_y)

    def compile(self, optimizer, loss, accuracy):
        self.optimizer = optimizer
        self.loss = loss
        self.accuracy = accuracy