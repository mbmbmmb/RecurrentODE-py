"""Port of aft/adjoint_odesystem_func.m (vectorised N-subject adjoint ODE)."""
from __future__ import annotations

import numpy as np

from ..common import spcol_deriv, unique_sort_index


def adjoint_odesystem_func(y_flat, theta, x_coef, time, knots, k):
    x_coef = np.asarray(x_coef, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    q = len(theta)
    N = x_coef.size
    # MATLAB reshapes y to (2+q, N); here each subject has q+2 state vars.
    y = np.asarray(y_flat, dtype=float).reshape(q + 2, N, order='F')
    u, bin_idx = unique_sort_index(y[0])
    B_u, dB_u = spcol_deriv(knots, k, u)
    B = B_u[bin_idx]
    dB = dB_u[bin_idx]
    res_1 = np.exp(B @ theta) * x_coef * time
    res_2 = -y[1] * res_1 * (dB @ theta)
    res_4 = -y[1][:, None] * res_1[:, None] * B
    return np.concatenate([[res_1], [res_2], res_4.T], axis=0).ravel(order='F')
