# NeuralNetworkScratch/models/model.py
import numpy as np
from tqdm import tqdm

from ..layers.base import Layer
from ..losses.base import Loss
from ..optimizers.base import Optimizer


class Model:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
        self.initializeWeights()

        self.optimizer = Optimizer()
        self.loss = Loss()

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

    def backward(self, output, y_ture):
        pass

    def fit(self, data, epochs):
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Please set an optimizer before calling fit().")
        if self.loss is None:
            raise ValueError("Loss not set. Please set an loss before calling fit().")

        for epoch in tqdm(range(epochs)):
            for batch_x, batch_y in data:
                last_layer = self.forward(batch_x)

                loss = self.loss.forward(last_layer, batch_y)
                # print('Output:', neurons[0])
                # print('Loss:', self.loss(batch_y, neurons[0]))
                d_values = self.loss.backward(last_layer, batch_y)
                for layer in reversed(self.layers[1:]):
                    d_values = layer.backward(d_values)
                    self.optimizer.update(layer)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss