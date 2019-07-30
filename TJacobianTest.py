import numpy as np
import matplotlib.pyplot as plt

def JacobianTest(Layer, X, JacMV):
    x_out = Layer.forward(X)

    epsilons = np.geomspace(1 / 1000000, 1 / 100, num=50)

    d = np.random.randn(X.shape[0], X.shape[1])

    e = []
    diffe = []
    diffee = []
    diffratio = []
    for epsilon in epsilons:
        e.append(epsilon)

        x_oute = Layer.forward(X + epsilon * d)

        diffe.append(np.linalg.norm(x_oute - x_out))

        diffee.append(np.linalg.norm(x_oute - x_out - JacMV(X, Layer, epsilon * d)))
    # plt.plot(e,diffratio,label=" O(e) vs O(e2)")
    plt.loglog(e, diffe, label="lin")
    plt.loglog(e, diffee, label="square")
    title = "jacobian test X"
    plt.title(title)
    #   plt.yticks(np.arange(min(diff_ratio), max(diff_ratio), 0.1))
    plt.legend()
    plt.show()