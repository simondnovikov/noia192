import numpy as np
from FLayer import Layer

class SoftMax(Layer):
    def __init__(self, num_channels, num_classes):
        # W is (number of channels+1,  number of classes)
        Layer.__init__(self, num_channels + 1, num_classes)
        self.IN = self.IN - 1

    def forward(self, X):
        # X is (chan_in,samples)
        # out is pmat is (chan_out,samples)
        # add one channel to

        Xb = np.ones((X.shape[0] + 1, X.shape[1]));
        Xb[:-1, :] = X  # (chan_in+1,samples)

        WTWB = Xb.T.dot(self.weights[0])  # (number of samples ,number of classes)

        normalizing = np.max(WTWB, axis=1)  # (number of samples, )

        WTWBN = np.subtract(WTWB.T, normalizing).T  # (number of samples ,number of classes)
        eXTwb = np.exp(WTWBN)  # (number of samples ,number of classes)

        sums = np.sum(eXTwb, axis=1)

        pmat = np.divide(eXTwb.T, sums)
        self.Xb = Xb

        return pmat

    def backward(self, G):
        # G is (chan_out,samples)
        # self.gradient is (chan_in+1,chan_out)
        # return is (chan_in,samples)
        S = G.shape[1]
        gradient = self.Xb.dot(G.T)

        self.gradients = [gradient]

        W = self.weights[0][:-1, :]
        # self.jacobian=sparse.kron(W,np.eye(S))

        # return np.reshape(self.jacobian.dot(G.ravel()), (self.IN,S))/S
        return W.dot(G)