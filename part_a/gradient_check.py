"""Numerical gradient check (Problem 2e). Compares analytic vs central finite differences."""
import numpy as np
from layers import Network

rng = np.random.default_rng(1)
X = rng.standard_normal((20, 2))
y = rng.integers(0, 3, 20)
net = Network([2, 5, 3], seed=0)

net.forward(X, y)
net.backward()                                   # analytic gradients are now stored in dW, db

EPS, worst = 1e-5, 0.0
lines = []
for li, d in enumerate(net.dense):
    for name, param, grad in [('W', d.W, d.dW), ('b', d.b, d.db)]:
        for _ in range(5):                       # 5 random entries per parameter tensor
            idx = tuple(rng.integers(0, s) for s in param.shape)
            orig = param[idx]
            param[idx] = orig + EPS; Lp = net.forward(X, y)
            param[idx] = orig - EPS; Lm = net.forward(X, y)
            param[idx] = orig
            g_num = (Lp - Lm) / (2 * EPS)
            g_ana = grad[idx]
            rel = abs(g_ana - g_num) / (abs(g_ana) + abs(g_num) + 1e-12)
            worst = max(worst, rel)
            lines.append(f'layer{li+1}.{name}{idx}: analytic={g_ana:+.8f} numeric={g_num:+.8f} rel_err={rel:.2e}')

print('\n'.join(lines))
msg = f'\nMax relative error = {worst:.2e} -> ' + ('PASS (< 1e-6)' if worst < 1e-6 else 'FAIL')
print(msg)
with open('results/tables/gradient_check.txt', 'w') as f:
    f.write('\n'.join(lines) + msg + '\n')