import numpy as np

def complex_harmonic_data(noise=0, space=None):
    if space is None:
        space = [-2 * np.pi, 4 * np.pi]

    dim = 300
    X = np.linspace(*space, dim).reshape(-1, 1)
    y = np.sin(X) + 1/2 * np.sin(3*X) - 2 * np.cos(2*X) + np.random.randn(dim, 1) * noise
    return X, y

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X, y = complex_harmonic_data()
    plt.plot(X, y)
    plt.show()
