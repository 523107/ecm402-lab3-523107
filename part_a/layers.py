"""DenseLayer, ReLU, SoftmaxCrossEntropy (Problem 2a-c). NumPy only."""
import numpy as np


class DenseLayer:
    def __init__(self, n_in, n_out, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        # He initialisation: W ~ N(0, 2/n_in). Keeps activation variance stable through ReLU layers.
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((1, n_out))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X):
        self.X = X                      # cache input for backward
        return X @ self.W + self.b

    def backward(self, dz):
        self.dW = self.X.T @ dz         # dL/dW
        self.db = dz.sum(axis=0, keepdims=True)   # dL/db
        return dz @ self.W.T            # dL/dX, passed to the previous block


class ReLU:
    def forward(self, z):
        self.mask = z > 0
        return z * self.mask

    def backward(self, da):
        return da * self.mask


class SoftmaxCrossEntropy:
    """Fused softmax + mean categorical cross-entropy. y = integer class labels."""
    def forward(self, z, y):
        z = z - z.max(axis=1, keepdims=True)         # stability: exp of large numbers overflows
        e = np.exp(z)
        self.p = e / e.sum(axis=1, keepdims=True)
        self.y = y
        n = len(y)
        return -np.mean(np.log(self.p[np.arange(n), y] + 1e-12))

    def backward(self):
        n = len(self.y)
        g = self.p.copy()
        g[np.arange(n), self.y] -= 1.0               # y_hat - y  (y is one-hot)
        return g / n                                  # Equation (2)


class Network:
    """Chains blocks: Dense -> ReLU -> Dense -> SoftmaxCrossEntropy (any depth)."""
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.dense, self.acts = [], []
        for i in range(len(sizes) - 1):
            self.dense.append(DenseLayer(sizes[i], sizes[i + 1], rng))
            if i < len(sizes) - 2:
                self.acts.append(ReLU())
        self.loss = SoftmaxCrossEntropy()

    def logits(self, X):
        for i, d in enumerate(self.dense):
            X = d.forward(X)
            if i < len(self.acts):
                X = self.acts[i].forward(X)
        return X

    def forward(self, X, y):
        return self.loss.forward(self.logits(X), y)

    def backward(self):
        g = self.loss.backward()
        for i in reversed(range(len(self.dense))):
            g = self.dense[i].backward(g)
            if i > 0:
                g = self.acts[i - 1].backward(g)

    def predict(self, X):
        return self.logits(X).argmax(axis=1)
        