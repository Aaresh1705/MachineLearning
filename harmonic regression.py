import mycelium as net
import numpy as np
import matplotlib.pyplot as plt

net.seed(42)


def plot_regression(X, y, yy):
    plt.plot(X, y, c='red', label='data')
    plt.plot(X, yy, c='blue', label='model')
    plt.legend(loc='upper right')
    plt.show()


sigmoid = net.activation.Sigmoid
relu = net.activation.Relu
sin = net.activation.Sin


def main():
    model = net.models.Model([
        net.layers.Input(2),
        net.layers.Dense(3, sin),
        net.layers.Dense(1)
    ])

    model.compile(
        optimizer=net.optimizers.Adam(),
        loss=net.losses.MSE(),
        accuracy=net.accuracy.MAE()
    )

    X, y = net.data.complex_harmonic_data(0.1)

    X_feat = np.concatenate([
        np.sin(X),
        np.cos(X),
    ], axis=1)

    data = net.data.Data(X_feat, y,
                         split_type=net.data.RandomSplit,
                         test_split=0.2)

    train = net.data.Batch(data.train.X, data.train.y)
    test = net.data.Batch(data.test.X, data.test.y)

    model.fit(train, test, 30000)

    yy = model.forward(X_feat)

    plot_regression(X, y, yy)

    X_wide, y_wide = net.data.complex_harmonic_data(0.1, space=[-4 * np.pi, 10 * np.pi])
    X_wide_feat = np.concatenate([
        np.sin(X_wide),
        np.cos(X_wide),
    ], axis=1)

    yy = model.forward(X_wide_feat)
    plot_regression(X_wide, y_wide, yy)


if __name__ == '__main__':
    main()

