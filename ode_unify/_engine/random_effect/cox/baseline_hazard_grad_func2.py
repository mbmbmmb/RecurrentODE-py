"""Port of random effect/cox/baseline_hazard_grad_func2.m.

Second-order gradient, flattened to a ``(len(time), q^2)`` matrix to match
the MATLAB convention used by downstream assembly code.
"""
from __future__ import annotations

import numpy as np

from ...common import spcol


def baseline_hazard_grad_func2(time, theta, knots, k):
    time = np.atleast_1d(np.asarray(time, dtype=float)).ravel()
    q = theta.size
    B = spcol(knots, k, time)
    BtB = (B.T @ B).reshape(1, q * q)
    return np.exp(B @ theta)[:, None] * BtB
