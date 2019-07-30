import numpy as np


class Layer():  # linear layer, to demonstrate that dimensions work/ show layer
    def __init__(self, num_channels, num_classes):
        W = np.random.randn(num_channels, num_classes)
        self.gradient = None
        self.X = None
        self.weights = [W]
        self.gradients = [self.gradient]
        self.jacobian = None
        self.orig_weights = None
        self.perturb_dirs = None
        self.prev_updates = None
        self.IN = num_channels
        self.OUT = num_classes

    def forward(self, X):
        # X is (chan_in,samples)
        # out is WX is (chan_out,samples)
        self.X = np.copy(X)
        return self.weights[0].T.dot(X)

    def backward(self, G):
        # G is (chan_out,samples)
        # self.gradient is (chan_in,chan_out)
        # return is (chan_in,samples)

        S = G.shape[1]
        gradient = self.X.dot(G.T)
        self.gradients = [gradient]
        # jacobian is  (inXsamples, outXsamples)
        # ravel the grad_in, then reshape back

        # self.jacobian=sparse.kron(self.weights[0],np.eye(S))     #much more complicated than it needs to be, to make dimensions work in more complex cases
        # return np.reshape(self.jacobian.dot(G.ravel()), (self.IN,S))/S
        return self.weights[0].dot(G)

    def update(self, lr, momentum, gamma=0):  # momentum, regularization
        if self.prev_updates is None:
            self.prev_updates = []
            for i in range(len(self.gradients)):
                self.prev_updates.append(lr * self.gradients[i])
        else:
            for i in range(len(self.gradients)):
                self.prev_updates[i] = momentum * self.prev_updates[i] + lr * self.gradients[i]
        for i in range(len(self.gradients)):
            self.weights[i] = self.weights[i] * (1 - gamma) - self.prev_updates[i]