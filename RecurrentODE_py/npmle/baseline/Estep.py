"""Port of npmle/baseline/Estep.m.

Compute posterior weights for the random effects on the Gauss-Hermite grid
and the scaled quadrature nodes ``Tbb = sqrtm(Sigma) @ bb.T``.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm

from .Gfun import Gfun


def Estep(ID, Y, XX, ZZ, Delta, n, oldbeta, oldf, oldSigma, bb, ww,
          indY, indID, rho, r1, r2):
    oldSigma_arr = np.atleast_2d(np.asarray(oldSigma, dtype=float))
    Tbb = sqrtm(oldSigma_arr) @ np.asarray(bb, dtype=float).T
    Tbb = np.real(Tbb)

    NN = Y.size
    mb = Tbb.shape[1]
    F = (indY @ oldf).reshape(-1, 1)
    EXZ = np.exp((XX @ oldbeta).reshape(-1, 1) + ZZ @ Tbb)
    G0, G1, G2, G3 = Gfun(EXZ * F, rho, r1, r2)

    temp = ZZ @ Tbb + np.log(G1)
    cens_mask = (Delta == 0)
    temp[cens_mask, :] = -G0[cens_mask, :]
    weight = np.exp(indID @ temp) * np.asarray(ww).reshape(1, -1)
    row_sums = weight.sum(axis=1, keepdims=True)
    weight = weight / row_sums
    return Tbb, weight
