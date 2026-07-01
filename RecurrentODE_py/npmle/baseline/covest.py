"""Port of npmle/baseline/covest.m.

Louis-style observed-information covariance estimator combining the
complete-data Hessian with the missing-information score-product.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm

from .Gfun import Gfun
from .scoregamma import scoregamma


def covest(ID, Y, XX, ZZ, Delta, n, newf, newbeta, newSigma, bb, weight,
           indY, indID, rho, r1, r2):
    XX = np.asarray(XX, dtype=float)
    ZZ = np.asarray(ZZ, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    Delta = np.asarray(Delta, dtype=float).ravel()
    ID = np.asarray(ID, dtype=int).ravel()
    newbeta = np.asarray(newbeta, dtype=float).ravel()
    newSigma_arr = np.atleast_2d(np.asarray(newSigma, dtype=float))
    bb = np.asarray(bb, dtype=float)

    newbb = np.real(sqrtm(newSigma_arr) @ bb.T)
    nbeta = newbeta.size - 1
    nsig = newSigma_arr.shape[0]
    ngamma = nsig * (nsig + 1) // 2

    nn = int(np.sum(Delta == 1))
    NN = Y.size
    mb = bb.shape[0]

    newXX = XX[:, 1:]
    newlambda = newf * np.exp(newbeta[0])
    newLambda = (indY @ newlambda).reshape(-1, 1)
    EXZ = np.exp((newXX @ newbeta[1:]).reshape(-1, 1) + ZZ @ newbb)
    G0, G1, G2, G3 = Gfun(EXZ * newLambda, rho, r1, r2)
    EG1 = G1 * EXZ
    EG2 = G2 * EXZ ** 2
    EG2G1 = G2 / G1 * EXZ
    EDG2G1 = (G3 / G1 - G2 ** 2 / G1 ** 2) * EXZ ** 2

    coef1 = EG2G1.copy();  coef1[Delta == 0, :] = -EG1[Delta == 0, :]
    coef2 = EDG2G1.copy(); coef2[Delta == 0, :] = -EG2[Delta == 0, :]
    c12 = coef1 + coef2 * newLambda

    npar = nbeta + ngamma
    if nsig > 1:
        tempDgamma, tempD2gamma = scoregamma(newSigma_arr, ngamma, newbb)
    else:
        # Scalar sigma fast path (matches main.m when Sigma0=1).
        sig = float(newSigma_arr[0, 0])
        tempDgamma = newbb ** 2 / (2.0 * sig ** 2) - 1.0 / (2.0 * sig)
        tempD2gamma = -newbb ** 2 / (sig ** 3) + 1.0 / (2.0 * sig ** 2)

    OY = np.sort(Y[Delta == 1])

    Dbeta2 = np.zeros((nbeta, nbeta))
    DbetaLambda = np.zeros((nbeta, nn))
    DLambda2 = np.zeros((nn, nn))
    Dgamma2 = np.zeros((ngamma, ngamma))
    DL = np.zeros((npar + nn, n))
    DL2 = np.zeros((npar + nn, npar + nn))

    for i in range(1, n + 1):
        mask = (ID == i)
        TX = newXX[mask, :]
        TLambda = newLambda[mask, 0]
        Tcoef1 = coef1[mask, :]
        Tcoef2 = coef2[mask, :]
        ETcoef2 = Tcoef2 @ weight[i - 1, :]
        Tc12 = c12[mask, :]
        ETc12 = Tc12 @ weight[i - 1, :]
        TY = Y[mask]
        TDelta = Delta[mask]
        ni = int(mask.sum())

        tempDbeta = np.zeros((nbeta, mb))
        tempDLambda = np.zeros((nn, mb))
        for j in range(ni):
            tempDbeta = (
                tempDbeta
                + TDelta[j] * np.broadcast_to(TX[j, :][:, None], (nbeta, mb))
                + TLambda[j] * TX[j, :][:, None] * Tcoef1[j, :][None, :]
            )
            indicator = (TY[j] >= OY).astype(float).reshape(-1, 1)
            tempDLambda = tempDLambda + indicator @ Tcoef1[j:j + 1, :]
            if TDelta[j] == 1:
                pos = int(np.where(OY == TY[j])[0][0])
                tempDLambda[pos, :] = (
                    tempDLambda[pos, :] + 1.0 / newlambda[pos]
                )
            Dbeta2 = Dbeta2 + ETc12[j] * TLambda[j] * np.outer(TX[j, :], TX[j, :])
            DbetaLambda = DbetaLambda + ETc12[j] * np.outer(
                TX[j, :], (TY[j] >= OY).astype(float),
            )
            rowind = (TY[j] >= OY).astype(float)
            DLambda2 = DLambda2 + ETcoef2[j] * np.outer(rowind, rowind)
            if TDelta[j] == 1:
                pos = int(np.where(OY == TY[j])[0][0])
                DLambda2[pos, pos] -= 1.0 / newlambda[pos] ** 2

        # scoregamma path (nsig>1) expects tempDgamma shape (ngamma, mb)
        if nsig > 1:
            Dgamma2 = Dgamma2 + (tempD2gamma @ weight[i - 1, :]).reshape(
                ngamma, ngamma,
            )
            temp = np.vstack([tempDbeta, tempDgamma, tempDLambda])
        else:
            Dgamma2 = Dgamma2 + float(tempD2gamma @ weight[i - 1, :])
            temp = np.vstack([tempDbeta, tempDgamma.reshape(1, mb), tempDLambda])

        DL[:, i - 1] = temp @ weight[i - 1, :]
        DL2 = DL2 + temp @ (np.broadcast_to(weight[i - 1, :], (npar + nn, mb)) * temp).T

    tempD2 = np.block([
        [Dbeta2,                          np.zeros((nbeta, ngamma)),  DbetaLambda],
        [np.zeros((ngamma, nbeta)),       Dgamma2,                    np.zeros((ngamma, nn))],
        [DbetaLambda.T,                   np.zeros((nn, ngamma)),     DLambda2],
    ])
    cov = -tempD2 - (DL2 - DL @ DL.T)
    return np.linalg.inv(cov)
