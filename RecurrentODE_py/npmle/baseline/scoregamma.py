"""Port of npmle/baseline/scoregamma.m.

Score/information contributions for the random-effects covariance
parameter ``gamma``.  Only used when ``nsig > 1``; in the reference
simulations ``Sigma`` is a scalar so this function is never actually
invoked by the shipped driver.

The MATLAB source contains two undefined-variable bugs
(``temp2 = temp.*cm`` and ``temp3(row+1, ...)``) that MATLAB silently
tolerates when the branch is dead.  The Python port mirrors the intent
(assuming ``temp`` should be ``temp2`` and ``row`` should be ``rowind``)
so that the vector-Sigma path is usable.
"""
from __future__ import annotations

import numpy as np


def scoregamma(newSigma, ngamma, Tbb):
    newSigma = np.atleast_2d(np.asarray(newSigma, dtype=float))
    nsig = newSigma.shape[0]
    mb = Tbb.shape[1]
    cm = 2.0 * np.ones((nsig, nsig)) - np.eye(nsig)
    inv_sig = np.linalg.inv(newSigma)

    tempDgamma = np.zeros((ngamma, mb))
    tempD2gamma = np.zeros((ngamma ** 2, mb))

    for k in range(mb):
        tempTbb = Tbb[:, k:k + 1]
        temp1 = inv_sig @ (tempTbb @ tempTbb.T) @ inv_sig / 2.0 - inv_sig / 2.0
        temp1 = temp1 * cm
        ind = 0
        for i in range(nsig):
            for j in range(i, nsig):
                tempDgamma[ind, k] = temp1[i, j]
                ind += 1

        temp3 = np.zeros((ngamma, ngamma))
        colind = 0
        for i in range(nsig):
            for j in range(i, nsig):
                E_ij = np.zeros((nsig, nsig))
                E_ij[i, j] = 1.0
                E_ij[j, i] = 1.0
                temp2 = (
                    inv_sig @ E_ij @ inv_sig @ (tempTbb @ tempTbb.T) @ inv_sig
                    + inv_sig @ (tempTbb @ tempTbb.T) @ inv_sig @ E_ij @ inv_sig
                ) / 2.0 + inv_sig @ E_ij @ inv_sig / 2.0
                temp2 = temp2 * cm
                rowind = 0
                for q in range(nsig):
                    for s in range(q, nsig):
                        temp3[rowind, colind] = temp2[q, s]
                        rowind += 1
                colind += 1

        tempind = 0
        for i in range(temp3.shape[0]):
            for j in range(i, temp3.shape[0]):
                tempD2gamma[tempind, k] = temp3[i, j]
                tempind += 1

    return tempDgamma, tempD2gamma
