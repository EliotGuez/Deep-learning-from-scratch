import numpy as np

"""
The pseudo-code for the optimizers comes from the book Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville.
"""


class Optimizer:

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, param, grad):
        pass


class SGD(Optimizer):

    def __init__(self, lr=0.01):
        super().__init__(lr=lr)

    def step(self, param, grad):
        param -= self.lr * grad
        return param


class SGD_Momentum(Optimizer):

    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr=lr)
        self.momentum = momentum
        self.v = None

    def step(self, param, grad):
        if self.v is None:
            self.v = np.zeros_like(grad)
        self.v = self.momentum * self.v - self.lr * grad
        param = param + self.v
        return param


class AdaGrad(Optimizer):

    def __init__(self, lr=0.01, epsilon=1e-7):
        super().__init__(lr=lr)
        self.epsilon = epsilon
        self.r = None

    def step(self, param, grad):
        if self.r is None:
            self.r = np.zeros_like(grad)
        self.r = self.r + grad ** 2
        param = param - (self.lr * grad) / (np.sqrt(self.r) + self.epsilon)
        return param


class RMSProp(Optimizer):
    
    def __init__(self, lr=0.01, rho=0.9, delta=1e-6):
        super().__init__(lr=lr)
        self.rho = rho
        self.delta = delta
        self.r = None

    def step(self, param, grad):
        if self.r is None:
            self.r = np.zeros_like(grad)

        self.r = self.rho * self.r + (1 - self.rho) * grad **2
        param = param - self.lr * grad / ((np.sqrt(self.r) + self.delta))
        return param


class Adam(Optimizer):

    def __init__(self, lr=0.001, rho1=0.9, rho2=0.999, epsilon=1e-8):
        super().__init__(lr=lr)
        self.rho1 = rho1
        self.rho2 = rho2
        self.epsilon = epsilon
        self.s = None
        self.r = None
        self.t = 0

    def step(self, param, grad):
        self.t += 1
        if self.s is None:
            self.s = np.zeros_like(grad)
            self.r = np.zeros_like(grad)
        self.s = self.rho1 * self.s + (1 - self.rho1) * grad
        self.r = self.rho2 * self.r + (1 - self.rho2) * grad * grad
        s_hat = self.s / (1 - self.rho1**self.t)
        r_hat = self.r / (1 - self.rho2**self.t)

        param -= self.lr * s_hat / (np.sqrt(r_hat) + self.epsilon)
        return param
