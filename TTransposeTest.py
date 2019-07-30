import numpy as np


def TransposeTest(Layer, X, JacMV):
    # purpose of test is to show correctness of JacMV, since the backward function we have is equivalent to JacTMV

    x_out = Layer.forward(X)

    for i in range(10):
        u = np.random.randn(x_out.shape[0], X.shape[1])
        v = np.random.randn(X.shape[0], X.shape[1])
        print(np.abs(np.sum(np.multiply(u, JacMV(X, Layer, v))) - np.sum(np.multiply(v, Layer.backward(u)))))
