import numpy as np

class Net():
    def __init__(self):
        self.layers = []
        self.lastX = None
        self.labels = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.lossfunction = loss

    def forward(self, X, labels=None):
        for i in range(len(self.layers)):
            X = np.copy(self.layers[i].forward(X))

        outlabel = np.argmax(X, axis=0)

        if labels is not None:
            loss = self.lossfunction.forward(X, labels)
            self.lastX = np.copy(X)
            self.labels = np.copy(labels)
            return loss, outlabel
        return X, outlabel  # For incomplete(without loss) networks

    def backward(self,
                 grad_back=None):  # For incomplete(without loss) networks, you have to provide your own grad back(jacobian test)
        if grad_back is None:
            grad_back = self.lossfunction.backward(self.lastX, self.labels)
        for i in reversed(range(len(self.layers))):
            grad_back = self.layers[i].backward(grad_back)
        return grad_back

    def update(self, lr, momentum, gamma=0):
        for i in range(len(self.layers)):
            self.layers[i].update(lr, momentum, gamma)

    ## for tests: adding epsilon*D to each weight
    def perturb_weights(self, epsilon):
        expected_change = 0
        for layer in self.layers:
            if layer.orig_weights is None:
                layer.orig_weights = []
                layer.perturb_dirs = []
                for weight in layer.weights:
                    layer.orig_weights.append(np.copy(weight))
                    layer.perturb_dirs.append(np.random.randn(weight.shape[0], weight.shape[1]))

            for i in range(len(layer.weights)):
                layer.weights[i] = layer.orig_weights[i] + epsilon * layer.perturb_dirs[i]
                expected_change = expected_change + epsilon * layer.perturb_dirs[i].ravel().dot(
                    layer.gradients[i].ravel())
        return expected_change

    def reset_perturb(self):
        for layer in self.layers:

            for i in range(len(layer.weights)):
                layer.weights[i] = layer.orig_weights[i]