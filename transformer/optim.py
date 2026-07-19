"""A minimal Adam optimizer that works over the params_and_grads() list
convention shared by every layer in this package."""
import numpy as np


class Adam:
    def __init__(self, params_and_grads_fn, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.get_pg = params_and_grads_fn
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for i, (p, g) in enumerate(self.get_pg()):
            if i not in self.m:
                self.m[i] = np.zeros_like(p)
                self.v[i] = np.zeros_like(p)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
