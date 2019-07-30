import numpy as np
import matplotlib.pyplot as plt

def GradientTestW(testedNet, X, labels,f):
    loss, outlabel = testedNet.forward(X, labels)
    gradient = testedNet.backward()
    epsilons = np.geomspace(1 / 1000000, 1 / 100, num=50)

    loss = np.copy(loss)
    e = []
    diffe = []
    diffee = []
    diffratio = []
    for epsilon in epsilons:
        e.append(epsilon)
        expected_change = testedNet.perturb_weights(epsilon)

        losse, outlabel = testedNet.forward(X, labels)

        diffe.append(np.abs(losse - loss))

        diffee.append(np.abs(losse - loss - expected_change))

        diffratio.append(diffe[-1] / diffee[-1])
        testedNet.reset_perturb()
    # plt.plot(e,diffratio,label=" O(e) vs O(e2)")
    plt.loglog(e, diffe, label="lin")
    plt.loglog(e, diffee, label="square")
    title = "gradient test W"
    plt.title(title)
    #   plt.yticks(np.arange(min(diff_ratio), max(diff_ratio), 0.1))
    plt.legend()
    filename = title + f + '.png'
    plt.savefig(filename, bbox_inches='tight')
