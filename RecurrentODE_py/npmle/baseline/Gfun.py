"""Port of npmle/baseline/Gfun.m.

G-transform and its first three derivatives for the Box-Cox family
(``rho > 0``) and the logarithmic family (``rho == 0``).
"""
from __future__ import annotations

import numpy as np


def Gfun(x, rho, r1, r2):
    x = np.asarray(x, dtype=float)
    if rho > 0:
        G0 = (np.power(1.0 + x, rho) - 1.0) / rho
        G1 = np.power(1.0 + x, rho - 1.0)
        G2 = (rho - 1.0) * np.power(1.0 + x, rho - 2.0)
        G3 = (rho - 1.0) * (rho - 2.0) * np.power(1.0 + x, rho - 3.0)
        return G0, G1, G2, G3
    if rho == 0:
        G0 = r1 * np.log1p(r2 * x)
        G1 = r1 * r2 / (1.0 + r2 * x)
        G2 = -r1 * r2 ** 2 / np.power(1.0 + r2 * x, 2)
        G3 = 2.0 * r1 * r2 ** 3 / np.power(1.0 + r2 * x, 3)
        return G0, G1, G2, G3
    raise ValueError(f'unsupported rho={rho}')
