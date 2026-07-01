"""Port of npmle/baseline/Mstep.m.

One Newton step for the complete-data parameters ``(beta, alpha)`` followed
by a recursion that re-solves for the baseline jumps ``f`` under the new
parameters.
"""
from __future__ import annotations

import numpy as np

from .Gfun import Gfun
from .recursive import recursive


def Mstep(ID, Y, XX, ZZ, Delta, n, oldalpha, oldbeta, oldSigma, Tbb, weight,
          indY, indID, rho, r1, r2):
    XX = np.asarray(XX, dtype=float)
    ZZ = np.asarray(ZZ, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    Delta = np.asarray(Delta, dtype=float).ravel()
    ID = np.asarray(ID, dtype=int).ravel()
    oldbeta = np.asarray(oldbeta, dtype=float).ravel()

    nbeta = oldbeta.size
    mz = Tbb.shape[0]
    newSigma = Tbb @ (np.broadcast_to(weight.mean(axis=0), (mz, weight.shape[1])) * Tbb).T

    NN = Y.size
    mb = Tbb.shape[1]
    Eweight = np.zeros((NN, mb))
    for i in range(1, n + 1):
        mask = (ID == i)
        Eweight[mask, :] = weight[i - 1, :]

    f, f1 = recursive(ID, Y, XX, ZZ, Delta, n, NN, mb, oldalpha, oldbeta,
                      Tbb, Eweight, rho, r1, r2)
    F = (indY @ f).reshape(-1, 1)
    F1 = indY @ f1  # (NN, nbeta+1)
    EXZ = np.exp((XX @ oldbeta).reshape(-1, 1) + ZZ @ Tbb)
    G0, G1, G2, G3 = Gfun(EXZ * F, rho, r1, r2)
    EG0 = (G0 * Eweight).sum(axis=1)
    EG1 = (G1 * EXZ * Eweight).sum(axis=1)
    EG2 = (G2 * EXZ ** 2 * Eweight).sum(axis=1)
    EG2G1 = (G2 / G1 * EXZ * Eweight).sum(axis=1)
    EDG2G1 = ((G3 / G1 - G2 ** 2 / G1 ** 2) * EXZ ** 2 * Eweight).sum(axis=1)

    coef1 = EDG2G1.copy(); coef1[Delta == 0] = -EG2[Delta == 0]
    coef2 = EG2G1.copy();  coef2[Delta == 0] = -EG1[Delta == 0]
    Fflat = F.ravel()
    c12 = coef2 + coef1 * Fflat

    Gamma1 = XX[Delta == 1, :].sum(axis=0) + (coef2 * Fflat) @ XX
    Gamma2 = f.sum() - 1.0
    XFX = XX * Fflat[:, None] + F1[:, 1:]
    Gamma1beta = XX.T @ (c12[:, None] * XFX)
    Gamma1alpha = XX.T @ (c12 * F1[:, 0])
    Gamma2beta = f1[:, 1:].sum(axis=0)
    Gamma2alpha = f1[:, 0].sum()
    HH = np.block([
        [Gamma1beta,            Gamma1alpha.reshape(-1, 1)],
        [Gamma2beta.reshape(1, -1), np.array([[Gamma2alpha]])],
    ])
    rhs = np.concatenate([Gamma1.ravel(), [Gamma2]])
    newpar = np.concatenate([oldbeta, [oldalpha]]) - np.linalg.solve(HH, rhs)

    newbeta = newpar[:nbeta]
    newalpha = newpar[nbeta]
    newf, newf1 = recursive(ID, Y, XX, ZZ, Delta, n, NN, mb, newalpha,
                            newbeta, Tbb, Eweight, rho, r1, r2)
    return newbeta, newf, newSigma
