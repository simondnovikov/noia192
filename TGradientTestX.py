import numpy as np
import matplotlib.pyplot as plt


def GradientTestX(testedNet, X, labels):
    loss, outlabel = testedNet.forward(X, labels)
    gradient = testedNet.backward()

    epsilons = np.geomspace(1 / 1000000, 1 / 100, num=50)

    d = np.random.randn(X.shape[0], X.shape[1])

    e = []
    diffe = []
    diffee = []
    diffratio = []
    for epsilon in epsilons:
        e.append(epsilon)

        losse, outlabel = testedNet.forward(X + epsilon * d, labels)[0:2]

        diffe.append(np.abs(losse - loss))

        diffee.append(np.abs(losse - loss - epsilon * np.sum(np.multiply(d, gradient))))
        diffratio.append(diffe[-1] / diffee[-1])
    # plt.plot(e,diffratio,label=" O(e) vs O(e2)")
    plt.loglog(e, diffe, label="lin")
    plt.loglog(e, diffee, label="square")
    title = "gradient test X"
    plt.title(title)
    #   plt.yticks(np.arange(min(diff_ratio), max(diff_ratio), 0.1))
    plt.legend()
    plt.show()