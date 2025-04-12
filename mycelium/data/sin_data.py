import numpy as np

def sin_data(noise=0):
    dim = 100
    X = np.linspace(0, 2*np.pi, dim).reshape(-1, 1)
    y = np.sin(X) + np.random.randn(dim, 1) * noise
    return X, y