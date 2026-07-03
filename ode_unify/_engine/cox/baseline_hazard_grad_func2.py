"""Port of cox/baseline_hazard_grad_func2.m.

Returns exp(B theta) * vec(B^T B) flattened to a row of length q^2 (as MATLAB
does with reshape(B'*B, 1, q^2)).
"""
from __future__ import annotations

import numpy as np

from ..common import spcol


def baseline_hazard_grad_func2(time, theta, knots, k):
    time = np.atleast_1d(np.asarray(time, dtype=float))
    q = len(theta)
    B = spcol(knots, k, time)
    outer = (B.T @ B).reshape(1, q * q, order='F')
    return np.exp(B @ theta)[:, None] * outer
