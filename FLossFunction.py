import numpy as np

class LossFunction():
    def forward(self ,X ,labels):
        # X is (classes,samples)
        # labels is (classes,samples)
        # return scalar

        return np.sum(np.abs( X -labels))

    def backward(self ,X ,labels):
        # X is (classes,samples)
        # labels is (classes,samples)
        # return (classes,samples)
        return ( X -labels ) < 0 -(( X -labels ) >0)