"""Port of npmle/minbeta.m.

Simple gradient descent with backtracking line search.  Kept available
because the MATLAB codebase provides it as an alternative optimizer; the
standard path still calls ``scipy.optimize.minimize`` in ``main.py``.
"""
from __future__ import annotations

import numpy as np


def minbeta(fun_beta, beta):
    beta_cur = np.asarray(beta, dtype=float).copy()
    l_prev = 0.0
    l_cur, grad = fun_beta(beta_cur)
    grad = np.asarray(grad, dtype=float).ravel()
    c = 0.4
    a = 0.8
    while abs((l_cur - l_prev) / (l_prev + 1.0)) > 1e-4:
        tau = 0.5
        while fun_beta(beta_cur - tau * grad)[0] > l_cur - c * tau * (grad @ grad):
            tau *= a
        beta_cur = beta_cur - tau * grad
        l_prev = l_cur
        l_cur, grad = fun_beta(beta_cur)
        grad = np.asarray(grad, dtype=float).ravel()
    return beta_cur
