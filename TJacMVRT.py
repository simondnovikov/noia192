import numpy as np


def JacMVRT(X,Layer,v): #JacMV for both Relu and tanh versions
    S=X.shape[1]
    W=Layer.weights[0][:-1,:]
    # return np.multiply(Layer.weights[0].T.dot(v),Layer.XWind)
    return np.multiply(W.T.dot(v),Layer.XWind)