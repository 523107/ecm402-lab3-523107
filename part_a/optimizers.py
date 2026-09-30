"""SGD with momentum, RMSProp, Adam (Problem 2d). Each has update(layer)."""
import numpy as np


class SGD:
    def __init__(self, lr=0.1, beta=0.9):
        self.lr, self.beta, self.state = lr, beta, {}

    def update(self, layer):
        s = self.state.setdefault(id(layer), dict(vW=np.zeros_like(layer.W), vb=np.zeros_like(layer.b)))
        s['vW'] = self.beta * s['vW'] - self.lr * layer.dW      # v <- beta v - eta g
        s['vb'] = self.beta * s['vb'] - self.lr * layer.db
        layer.W += s['vW']                                       # theta <- theta + v
        layer.b += s['vb']


class RMSProp:
    def __init__(self, lr=0.01, rho=0.9, eps=1e-8):
        self.lr, self.rho, self.eps, self.state = lr, rho, eps, {}

    def update(self, layer):
        s = self.state.setdefault(id(layer), dict(sW=np.zeros_like(layer.W), sb=np.zeros_like(layer.b)))
        s['sW'] = self.rho * s['sW'] + (1 - self.rho) * layer.dW ** 2
        s['sb'] = self.rho * s['sb'] + (1 - self.rho) * layer.db ** 2
        layer.W -= self.lr * layer.dW / (np.sqrt(s['sW']) + self.eps)
        layer.b -= self.lr * layer.db / (np.sqrt(s['sb']) + self.eps)


class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.state = lr, beta1, beta2, eps, {}

    def update(self, layer):
        s = self.state.setdefault(id(layer), dict(
            mW=np.zeros_like(layer.W), sW=np.zeros_like(layer.W),
            mb=np.zeros_like(layer.b), sb=np.zeros_like(layer.b), t=0))
        s['t'] += 1
        t = s['t']
        for p, g, m, v in [(layer.W, layer.dW, 'mW', 'sW'), (layer.b, layer.db, 'mb', 'sb')]:
            s[m] = self.b1 * s[m] + (1 - self.b1) * g
            s[v] = self.b2 * s[v] + (1 - self.b2) * g ** 2
            m_hat = s[m] / (1 - self.b1 ** t)          # bias correction
            v_hat = s[v] / (1 - self.b2 ** t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)   # in-place update