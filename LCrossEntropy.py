import numpy as np
from FLossFunction import LossFunction

class crossentropy(LossFunction):
    def forward(self, pmat, labels):
        # X is (classes,samples)
        # labels is (classes,samples)
        # return (classes,samples)
        S = pmat.shape[1]
        loss = -np.sum(np.multiply(np.log(pmat + 0.0000001), labels)) / S

        return loss

    def backward(self, pmat, labels):
        S = pmat.shape[1]
        # X is (classes,samples)
        # labels is (classes,samples)
        # return (classes,samples)
        return (pmat - labels) / S