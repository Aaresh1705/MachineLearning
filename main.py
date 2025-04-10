from matplotlib.pyplot import legend

import NeuralNetworkScratch as Net
import numpy as np
import matplotlib.pyplot as plt
import time
import nnfs
from nnfs.datasets import sine_data

import test

nnfs.init()


def plot_regression(X, y, c, label):
    plt.plot(X, y, c)
    plt.legend([label], loc='upper right')


sigmoid = Net.activation.sigmoid.Sigmoid
relu = Net.activation.relu.Relu
sin = Net.activation.sinus.Sin


def main():
    model = Net.models.Model([
        Net.layers.Input(1),
        Net.layers.Dense(64, relu),
        Net.layers.Dense(1)
    ])

    model.compile(
        optimizer=Net.optimizers.SGD(0.0001),
        loss=Net.losses.MSE(),
    )
    dim = 25

    x = np.linspace(0, 5, dim).reshape((dim, 1))
    y = np.sin(x).reshape((dim, 1))
    plot_regression(x, y, c='red', label='sinus')

    t_start = time.time()
    model.fit(x, y, 10000)
    t_stop = time.time()
    print(f'It took {t_stop - t_start} seconds')

    with open('test.csv', 'a') as file:
        file.write(str(t_stop - t_start) + '\n')

    yy = []
    for x_ in x:
        yy.append(model.forward(x_))

    plot_regression(x, yy, c='blue', label='neural network')
    plt.show()


def version_2():
    model = Net.models.Model([
        Net.layers.Input(1),
        Net.layers.Dense(50, relu),
        Net.layers.Dense(1)
    ])

    model.compile(
        optimizer=Net.optimizers.Adam(),
        loss=Net.losses.MSE()
    )

    dim = 25
    X, y = sine_data()
    #x = np.linspace(0, 5, dim).reshape((dim, 1))
    #y = np.sin(x).reshape((dim, 1))
    data = Net.data.Batch(X, y, 16)
    model.fit(data, 10000)

    plt.plot(X, y, c='red')
    yy = model.forward(X)
    plt.plot(X, yy, c='blue')
    plt.show()


if __name__ == '__main__':
    version_2()

