import numpy as np
class loss:
    def forward(self, A, Y): pass
    def backward(self): pass


class CrossEntropy(loss):
    def __init__(self, num_classes):
        self._num_classes = num_classes

    def update(self, lr):
        pass

    def forward(self, A, Y):
        m = Y.size
        log_likelihood = -np.log(A[Y, np.arange(m)] + 1e-9)
        return np.sum(log_likelihood) / m

    def backward(self, A, Y):
        m = Y.size
        one_hot_Y = self.one_hot(Y)
        # Combined derivative of Softmax + CrossEntropy
        return A - one_hot_Y.T

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, self._num_classes))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y