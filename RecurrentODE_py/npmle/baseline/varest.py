"""Port of npmle/baseline/varest.m.

Alternative variance estimator via empirical score product.  Not exercised
by the reference driver (``main.py`` uses ``covest``) and the MATLAB source
contains several typos (``socre`` for ``score``, ``tempnbeta`` undefined).
This port fixes the obvious typos and assumes the intended vector ``beta``
has an intercept in the first slot.
"""
from __future__ import annotations

import numpy as np

from .Gfun import Gfun


def varest(ID, Y, XX, ZZ, Delta, n, f, beta, Sigma, bb, weight, indY, indID,
           rho, r1, r2):
    XX = np.asarray(XX, dtype=float)
    ZZ = np.asarray(ZZ, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    Delta = np.asarray(Delta, dtype=float).ravel()
    ID = np.asarray(ID, dtype=int).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    bb = np.asarray(bb, dtype=float)

    tempbeta = beta[1:]
    tempX = XX[:, 1:]
    Tbb = np.sqrt(float(Sigma)) * bb.T
    NN = Y.size
    mb = bb.shape[1] if bb.ndim > 1 else bb.shape[0]
    Eweight = np.zeros((NN, mb))
    for i in range(1, n + 1):
        Eweight[ID == i, :] = weight[i - 1, :]

    F = (indY @ f).reshape(-1, 1)
    EXZ = np.exp((XX @ beta).reshape(-1, 1) + ZZ @ Tbb)
    G0, G1, G2, G3 = Gfun(EXZ * F, rho, r1, r2)
    EG1 = (G1 * EXZ * Eweight).sum(axis=1)
    EG2 = (G2 * EXZ ** 2 * Eweight).sum(axis=1)
    EG2G1 = (G2 / G1 * EXZ * Eweight).sum(axis=1)

    Lambda = float(np.exp(beta[0])) * F.ravel()
    nn = int(np.sum(Delta == 1))
    nx = tempbeta.size

    YY = Y.copy()
    YY[Delta == 0] = Y.max() + 1.0
    OY = np.sort(YY)
    order = np.argsort(YY)
    tempind = np.zeros(NN, dtype=int)
    tempind[order] = np.arange(NN)

    coef2 = EG2G1.copy()
    coef2[Delta == 0] = -EG1[Delta == 0]

    score = np.zeros((NN, nn + nx))
    score[:, :nx] = tempX + (coef2 * Lambda)[:, None] * tempX
    event_rows = np.where(Delta == 1)[0]
    for i in event_rows:
        score[i, nx + tempind[i]] = 1.0 / f[tempind[i]]
    score[:, nx:] = score[:, nx:] + coef2[:, None] * indY

    Tscore = np.zeros((n, nn + nx + 1))
    Tscore[:, :nn + nx] = indID @ score

    for i in range(1, n + 1):
        Ebb = bb @ (np.broadcast_to(weight[i - 1, :], (1, mb)) * bb).T
        psibb = np.linalg.inv(np.atleast_2d(Sigma)) @ Ebb @ np.linalg.inv(np.atleast_2d(Sigma)) / 2.0 \
                - np.linalg.inv(np.atleast_2d(Sigma)) / 2.0
        Tscore[i - 1, nn + nx:] = float(psibb)

    return np.linalg.inv(Tscore.T @ Tscore)
