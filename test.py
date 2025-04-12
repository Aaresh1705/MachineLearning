import mycelium as net

if __name__ == '__main__':
    net.seed(42)

    X, y = net.data.sin_data()

    Relu = net.activation.Relu
    Sin = net.activation.Sin

    dense1 = net.layers.Input(1)
    dense2 = net.layers.Dense(500, Relu)
    dense3 = net.layers.Dense(1)

    dense2.initializeWeights(dense1.dim)
    dense3.initializeWeights(dense2.dim)

    loss_function = net.losses.MSE()
    optimizer = net.optimizers.Adam()
    accuracy = net.accuracy.MAE()

    for epoch in range(10001):
        hidden1 = dense1.forward(X)
        hidden2 = dense2.forward(hidden1)
        hidden3 = dense3.forward(hidden2)

        data_loss = loss_function.forward(hidden3, y)
        acc = accuracy.forward(hidden3, y)

        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
                  f'acc: {acc:.3f}, ' +
                  f'loss: {data_loss:.3f} (' +
                  f'lr: {optimizer.current_learning_rate}')

        d_value3 = loss_function.backward(hidden3, y)
        d_value2 = dense3.backward(d_value3)
        d_value1 = dense2.backward(d_value2)

        optimizer.pre_update_params()
        optimizer.update(dense2)
        optimizer.update(dense3)
        optimizer.post_update_params()

    import matplotlib.pyplot as plt

    hidden1 = dense1.forward(X)
    hidden2 = dense2.forward(hidden1)
    hidden3 = dense3.forward(hidden2)

    plt.plot(X, y, label='data')
    plt.plot(X, hidden3, label='model')
    plt.legend()
    plt.show()