"""Port of npmle/baseline/recursive.m.

Solves the NPMLE self-consistency recursion for the jumps ``f`` of the
baseline cumulative hazard at the ordered unique event times, together with
derivatives ``f1`` w.r.t. ``(alpha, beta)``.
"""
from __future__ import annotations

import numpy as np

from .Gfun import Gfun


def recursive(ID, Y, XX, ZZ, Delta, n, NN, mb, alpha, beta, Tbb, Eweight,
              rho, r1, r2):
    XX = np.asarray(XX, dtype=float)
    ZZ = np.asarray(ZZ, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    Delta = np.asarray(Delta, dtype=float).ravel()
    beta = np.asarray(beta, dtype=float).ravel()

    EXZ = np.exp((XX @ beta).reshape(-1, 1) + ZZ @ Tbb)
    nn = int(np.sum(Delta == 1))
    nbeta = beta.size

    f = np.zeros(nn)
    f1 = np.zeros((nn, nbeta + 1))

    OY = np.sort(Y[Delta == 1])

    f[nn - 1] = alpha
    f1[nn - 1, 0] = 1.0
    for k in range(2, nn + 1):
        j = nn - k  # 0-based index for time point j+1 in MATLAB
        Fj = 1.0 - f[j + 1:].sum()
        Fj1 = -np.ones((1, nn - (j + 1))) @ f1[j + 1:, :]  # (1, nbeta+1)

        Cind = (OY[j] <= Y) & (OY[j + 1] > Y)
        sel = np.where(Cind)[0]
        TDelta = Delta[sel]
        nj = sel.size
        if nj == 0:
            # Degenerate case: treat temp integrals as zero.
            tempf = 1.0 / f[j + 1]
            tempfalpha = -f1[j + 1, 0] / f[j + 1] ** 2
            tempfbeta = -f1[j + 1, 1:] / f[j + 1] ** 2
            f[j] = 1.0 / tempf
            f1[j, 0] = -f[j] ** 2 * tempfalpha
            f1[j, 1:] = -f[j] ** 2 * tempfbeta
            continue
        Ewj = Eweight[sel, :]
        EXZj = EXZ[sel, :]
        G0, G1, G2, G3 = Gfun(EXZj * Fj, rho, r1, r2)
        EG0 = (G0 * Ewj).sum(axis=1)
        EG1 = (G1 * EXZj * Ewj).sum(axis=1)
        EG2 = (G2 * EXZj ** 2 * Ewj).sum(axis=1)
        EG2G1 = (G2 / G1 * EXZj * Ewj).sum(axis=1)
        EDG2G1 = ((G3 / G1 - G2 ** 2 / G1 ** 2) * EXZj ** 2 * Ewj).sum(axis=1)

        coef1 = -EG2G1
        coef1[TDelta == 0] = EG1[TDelta == 0]
        coef2 = -EDG2G1
        coef2[TDelta == 0] = EG2[TDelta == 0]

        tempf = 1.0 / f[j + 1] + coef1.sum()
        tempfalpha = -f1[j + 1, 0] / f[j + 1] ** 2 + coef2.sum() * Fj1[0, 0]
        Xsel = XX[sel, :]
        tempfbeta = (
            -f1[j + 1, 1:] / f[j + 1] ** 2
            + coef2 @ (Xsel * Fj + np.broadcast_to(Fj1[:, 1:], (nj, nbeta)))
            + coef1 @ Xsel
        )
        f[j] = 1.0 / tempf
        f1[j, 0] = -f[j] ** 2 * tempfalpha
        f1[j, 1:] = -f[j] ** 2 * tempfbeta

    return f, f1
