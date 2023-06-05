"""
The following functions are due to https://arxiv.org/abs/2010.07167
"""
import torch as t
from math import exp
import torch.nn as nn

class PfndELU(t.nn.Module):
    def __init__(self, dim=1, ls=16, n_condim=4):
        super(PfndELU, self).__init__()
        self.sig = t.nn.Sigmoid()
        self.log_jacobian = 0
        self.dim = dim
        self.ls = ls
        self.inviter = 10
        if n_condim > 0:
            self.con = True
            self.subnet = t.nn.Sequential(t.nn.Linear(n_condim, 50), t.nn.ReLU(), t.nn.Linear(50, 50), t.nn.ReLU(),
                                          t.nn.Linear(50, 3*dim*(ls+1)))
            self.getParas(t.zeros((1, n_condim)))
        else:
            self.con = False
            self.mat1 = t.nn.Parameter(t.randn(dim, ls), True)
            self.bias1 = t.nn.Parameter(t.randn(dim, ls), True)
            self.mat2 = t.nn.Parameter(t.randn(dim, ls), True)
            self.bias2 = t.nn.Parameter(t.randn(dim), True)
            self.eps = t.nn.Parameter(t.zeros(dim)-1, True)
            self.alpha = t.nn.Parameter(t.zeros(dim), True)


    def actfunc(self, x):
        return t.nn.ELU()(x)
        # return t.nn.ReLU()(x)

    def actderiv(self, x):
        return 1 + self.actfunc(-t.nn.ReLU()(-x))
        # return (x > t.zeros_like(x)).float()

    def forward(self, x, y=None, rev=False):
        if self.con:
            self.getParas(y)
        if not rev:
            self.log_jacobian = t.log(self.abl(x))
            return self.function(x)
        else:
            z = self.inv(x, n=self.inviter)
            self.log_jacobian = - t.log(self.abl(z))
            return z

    def function(self, x):
        xn = x.unsqueeze(2)
        if self.con:
            return t.exp(self.alpha) * (x + 0.8 * self.sig(self.eps) * t.sum(self.actfunc(xn * self.mat1 + self.bias1) * self.mat2 / (t.sum(t.nn.ReLU()(-self.mat1 * self.mat2), dim=2).unsqueeze(1)+1), dim=2) + self.bias2)
        else:
            return t.exp(self.alpha) * (x + 0.8 * self.sig(self.eps) * t.sum(self.actfunc(xn * self.mat1 + self.bias1) * self.mat2 / (t.sum(t.nn.ReLU()(-self.mat1 * self.mat2), dim=1).unsqueeze(1)+1), dim=2) + self.bias2)

    def abl(self, x):
        xn = x.unsqueeze(2)
        if self.con:
            return t.exp(self.alpha) * self.sig(self.eps) * (t.sum(0.8 * self.actderiv(xn * self.mat1 + self.bias1) * self.mat2 * self.mat1 / (t.sum(t.nn.ReLU()(-self.mat1 * self.mat2), dim=2).unsqueeze(1)+1), dim=2) + 1)
        else:
            return t.exp(self.alpha) * (t.sum(0.8 * self.sig(self.eps) * self.actderiv(xn * self.mat1 + self.bias1) * self.mat2 * self.mat1 / (t.sum(t.nn.ReLU()(-self.mat1 * self.mat2), dim=1).unsqueeze(1)+1), dim=2) + 1)

    def inv(self, y, n=10):
        yn = y*t.exp(-self.alpha)-self.bias2
        for i in range(n):
            yn = yn - (self.function(yn) - y) / (self.abl(yn))
        return yn

    def jacobian(self, x, rev=False):
        return self.log_jacobian

    def getParas(self, y):
        param = self.subnet(y)
        size = len(y)
        self.mat1 = t.reshape(param[:, :self.dim*self.ls], (size, self.dim, self.ls))
        self.bias1 = t.reshape(param[:, self.dim*self.ls:2*self.dim*self.ls], (size, self.dim, self.ls))
        self.mat2 = t.reshape(param[:, 2*self.dim*self.ls:3*self.dim*self.ls], (size, self.dim, self.ls))
        self.bias2 = t.reshape(param[:, 3*self.dim*self.ls:3*self.dim*self.ls+self.dim], (size, self.dim))
        self.eps = t.reshape(param[:, 3*self.dim*self.ls+self.dim:3*self.dim*self.ls+2*self.dim], (size, self.dim))/10.0
        self.alpha = t.reshape(param[:, 3*self.dim*self.ls+2*self.dim:], (size, self.dim))/10.0


class PfndTanh(t.nn.Module):
    def __init__(self, dim=1, ls=16, n_condim=4):
        super(PfndTanh, self).__init__()
        self.sig = t.nn.Sigmoid()
        self.log_jacobian = 0
        self.dim = dim
        self.ls = ls
        self.clamp = 5
        self.inviter = 10
        if n_condim > 0:
            self.con = True
            self.subnet = t.nn.Sequential(t.nn.Linear(n_condim, 50), t.nn.ReLU(), t.nn.Linear(50, 50), t.nn.ReLU(),
                                          t.nn.Linear(50, 3*dim*(ls+1)))
            self.getParas(t.zeros(1, n_condim))
        else:
            self.con = False
            self.mat1 = t.nn.Parameter(t.randn(dim, ls), True)
            self.bias1 = t.nn.Parameter(t.randn(dim, ls), True)
            self.mat2 = t.nn.Parameter(t.randn(dim, ls), True)
            self.bias2 = t.nn.Parameter(t.randn(dim), True)
            self.eps = t.nn.Parameter(t.zeros(dim)-1, True)
            self.alpha = t.nn.Parameter(t.zeros(dim), True)

    def forward(self, x, y=None, rev=False):
        if self.con:
            self.getParas(y)
        if not rev:
            self.log_jacobian = t.log(self.abl(x))
            return self.function(x)
        else:
            z = self.inv(x, n=self.inviter)
            self.log_jacobian = - t.log(self.abl(z))
            return z


    def actfunc(self, x):
        # return t.tanh(x)
        return t.exp(-x**2) * t.tensor(1.16)

    def actderiv(self, x):
        # return 1-self.actfunc(x)**2
        return -2*x*self.actfunc(x)

    def function(self, x):
        xn = x.unsqueeze(2)
        if self.con:
            return t.exp(self.alpha) * (x + 0.8 * self.sig(self.eps) * t.sum(self.actfunc(xn * self.mat1 + self.bias1) * self.mat2 / (t.sum(t.abs(self.mat1 * self.mat2), dim=2).unsqueeze(1)+1), dim=2) + self.bias2)
        else:
            return t.exp(self.alpha) * (x + 0.8 * self.sig(self.eps) * t.sum(self.actfunc(xn * self.mat1 + self.bias1) * self.mat2 / (t.sum(t.abs(self.mat1 * self.mat2), dim=1).unsqueeze(1)+1), dim=2) + self.bias2)


    def abl(self, x):
        xn = x.unsqueeze(2)
        if self.con:
            return t.exp(self.alpha) * self.sig(self.eps) * (0.8 * t.sum(self.actderiv(xn * self.mat1 + self.bias1) * self.mat2 * self.mat1 / (t.sum(t.abs(self.mat1 * self.mat2), dim=2).unsqueeze(1)+1), dim=2) + 1)
        else:
            return t.exp(self.alpha) * (t.sum(0.8 * self.sig(self.eps) * (self.actderiv(xn * self.mat1 + self.bias1)) * self.mat2 * self.mat1 / (t.sum(t.abs(self.mat1 * self.mat2), dim=1).unsqueeze(1)+1), dim=2) + 1)


    def inv(self, y, n=5):
        yn = y*t.exp(-self.alpha)-self.bias2
        for i in range(n):
            yn = yn - (self.function(yn) - y) / self.abl(yn)
        return yn

    def jacobian(self, x, rev=False):
        return self.log_jacobian

    def getParas(self, y):
        param = self.subnet(y)
        size = len(y)
        self.mat1 = t.reshape(param[:, :self.dim*self.ls], (size, self.dim, self.ls))
        self.bias1 = t.reshape(param[:, self.dim*self.ls:2*self.dim*self.ls], (size, self.dim, self.ls))
        self.mat2 = t.reshape(param[:, 2*self.dim*self.ls:3*self.dim*self.ls], (size, self.dim, self.ls))
        self.bias2 = t.reshape(param[:, 3*self.dim*self.ls:3*self.dim*self.ls+self.dim], (size, self.dim))
        self.eps = t.reshape(param[:, 3*self.dim*self.ls+self.dim:3*self.dim*self.ls+2*self.dim], (size, self.dim))/10.0
        self.alpha = t.reshape(param[:, 3*self.dim*self.ls+2*self.dim:], (size, self.dim))/10.0


class Net(t.nn.Module):
    def __init__(self, n_blocks=3, dim=1, ls=16, n_condim=4):
        super().__init__()
        self.n_blocks = n_blocks
        mods = []
        for i in range(n_blocks):
            mods.append(PfndTanh(dim=dim, ls=ls, n_condim=n_condim))
            mods.append(PfndELU(dim=dim, ls=ls, n_condim=n_condim))
        self.blocks = t.nn.ModuleList(mods)
        self.log_jacobian = t.zeros(dim)

    def forward(self, x, y=None, rev=False):
        self.log_jacobian = 0.
        if not rev:
            for block in self.blocks:
                x = block(x, y)
                self.log_jacobian += block.jacobian(x)
            return x
        else:
            for block in self.blocks[::-1]:
                x = block.forward(x, y, rev=True)
                self.log_jacobian += block.jacobian(x)
            return x

    def jacobian(self, x):
        return self.log_jacobian
