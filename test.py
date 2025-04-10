import numpy as np
import matplotlib.pyplot as plt
import time


def test_matrix_mult():
    N = 10000
    input_dim = 10
    output_dims = np.linspace(1, 1000, 100,  dtype=int)
    time_for_dim_py = []
    time_for_dim_npy = []
    for output_dim in output_dims:
        w = np.random.randn(output_dim, input_dim)
        b = np.random.randn(output_dim)
        a = np.random.uniform(-5, 5, (N, input_dim))
        start = time.time()
        for _ in a:
            w @ _ + b
        end = time.time()
        time_for_dim_py.append(end - start)
        start = time.time()
        for _ in a:
            np.dot(w, _) + b
        end = time.time()
        time_for_dim_npy.append(end - start)
    plt.plot(output_dims, time_for_dim_py, 'blue')
    plt.plot(output_dims, time_for_dim_npy, 'red')
    plt.legend(['Python matrix multiplication', 'Numpy matrix multiplication'])
    plt.show()