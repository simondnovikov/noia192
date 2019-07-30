import numpy as np
from FLayer import Layer

class NeuralLayerT(Layer):  # Tanh version
    def __init__(self, num_channels, num_classes):
        Layer.__init__(self, num_channels + 1, num_classes)
        self.IN = self.IN - 1

    def forward(self, X):
        # returns the loss, and the label for each sample.
        # labels are number of possible classes, 1 hot encoded X number of samples
        # X  are number of channels X number of samples
        m = X.shape[1]

        # add one channel to
        Xb = np.ones((X.shape[0] + 1, X.shape[1]))
        Xb[:-1, :] = X

        XW = self.weights[0].T.dot(Xb)  # (out_channels ,number of samples)
        XW = np.tanh(XW)
        self.Xb = np.copy(Xb)
        self.XWind = 1 - np.square(XW)  # tanh'=1-tanh
        return XW

    def backward(self, G):
        # Xb is (in+1,samples)
        # G is (out, samples)
        # grad_downstream is (in,samples)
        S = G.shape[1]
        gradient = self.Xb.dot(np.multiply(G, self.XWind).T)

        self.gradients = [gradient]

        W = self.weights[0][:-1, :]

        return W.dot(np.multiply(self.XWind, G))