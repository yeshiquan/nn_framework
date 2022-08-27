import numpy as np
from layers import Function

class Loss(Function):
    def forward(self, X, Y):
        pass

    def backward(self):
        return self.grad["X"]

    def local_grad(self, X, Y):
        pass


class MeanSquareLoss(Loss):
    def forward(self, X, Y):
        # calculating loss
        d = np.square(X - Y)
        mse_loss = (np.square(X - Y)).mean(axis=1)
        return mse_loss

    def local_grad(self, X, Y):
        grads = {"X": 2 * (X - Y) / X.shape[0]}
        return grads


class CrossEntropyLoss(Loss):
    def forward(self, X, y):
        # calculating crossentropy
        exp_x = np.exp(X)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        log_probs = -np.log([probs[i, y[i]] for i in range(len(probs))])
        crossentropy_loss = np.mean(log_probs)

        # caching for backprop
        self.cache["probs"] = probs
        self.cache["y"] = y

        return crossentropy_loss

    def local_grad(self, X, Y):
        probs = self.cache["probs"]
        ones = np.zeros_like(probs)
        for row_idx, col_idx in enumerate(Y):
            ones[row_idx, col_idx] = 1.0

        grads = {"X": (probs - ones) / float(len(X))}
        return grads