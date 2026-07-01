"""Port of npmle/baseline/main.m.

Runs a single replication of the NPMLE baseline (EM with Gauss-Hermite
quadrature on the random effect).  Output is written to
``res_generator/npmle_N<N>_seed<seed>_setting_<label>.mat`` with fields
``beta_hat``, ``se_beta``, and ``runtime``.
"""
from __future__ import annotations

import os
import time as _time
import numpy as np

from ...common import ensure_dir
from .Estep import Estep
from .Mstep import Mstep
from .covest import covest
from .generator_npmle import generator_npmle


def _gauss_hermite_nodes():
    # MATLAB uses the fixed 20-point Gauss-Hermite rule hard-coded in main.m
    # (10 nodes mirrored about zero; scaled by sqrt(2) to match the
    # exp(-x^2/2) density).  Return (ww, bb) in that same order.
    w_half = np.array([
        2.22939364554E-013, 4.39934099226E-010, 1.08606937077E-007,
        7.8025564785E-006,  0.000228338636017,  0.00324377334224,
        0.0248105208875,    0.10901720602,      0.286675505363,
        0.462243669601,
    ])
    b_half = np.array([
        -5.38748089001, -4.60368244955, -3.94476404012, -3.34785456738,
        -2.78880605843, -2.25497400209, -1.73853771212, -1.2340762154,
        -0.737473728545, -0.245340708301,
    ])
    ww = np.concatenate([w_half, w_half])
    bb = np.sqrt(2.0) * np.concatenate([b_half, -b_half])
    return ww.reshape(-1, 1), bb.reshape(-1, 1)


def main(n, seed, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    data_dir = os.path.join(root, 'data_generator')
    res_dir = os.path.join(root, 'res_generator')
    ensure_dir(data_dir); ensure_dir(res_dir)

    rho = 0.5; r1 = 1.0
    beta0 = np.array([1.0, 1.0, 1.0])
    Sigma0 = 1.0
    alpha0 = 0.2  # noqa: F841  (kept for parity with main.m docstring)

    nbeta = beta0.size
    ngamma = 1
    mz = 1
    tau = 4.0
    testpt = tau / np.array([4.0, 3.0, 2.0, 1.0])
    npt = testpt.size

    ww, bb = _gauss_hermite_nodes()
    epsilon = 5e-3
    maxiter = 100

    data_setting = 'NPMLE'
    generator_npmle(n, seed, data_setting, data_dir=data_dir)

    data_file = os.path.join(
        data_dir, f'simudata_N{n}_seed{seed}_setting_{data_setting}.npz',
    )
    data = np.load(data_file)
    ID = data['ID'].ravel().astype(int)
    X = data['X']
    Y = data['Y'].ravel()
    Delta = data['Delta'].ravel()
    rho1 = float(np.asarray(data['rho1']).ravel()[0])
    rho = rho1
    r2 = 1.0 / r1

    NN = Y.size
    XX = np.column_stack([np.ones(NN), X])
    ZZ = np.ones((NN, 1))
    nn = int(np.sum(Delta == 1))
    OY = np.sort(Y[Delta == 1])
    indY = (Y[:, None] >= OY[None, :]).astype(float)
    indID = np.zeros((n, NN))
    for i in range(1, n + 1):
        indID[i - 1, ID == i] = 1.0

    oldbeta = np.zeros(beta0.size + 1)  # intercept + p
    oldalpha = 1.0 / nn
    oldf = oldalpha + np.zeros(nn)
    oldSigma = np.eye(mz)

    error = 1.0
    iteration = 0
    t0 = _time.time()
    while error > epsilon and iteration < maxiter:
        Tbb, weight = Estep(
            ID, Y, XX, ZZ, Delta, n, oldbeta, oldf, oldSigma,
            bb, ww, indY, indID, rho, r1, r2,
        )
        newbeta, newf, newSigma = Mstep(
            ID, Y, XX, ZZ, Delta, n, oldalpha, oldbeta, oldSigma,
            Tbb, weight, indY, indID, rho, r1, r2,
        )
        diag_diff = np.diag(np.atleast_2d(newSigma) - np.atleast_2d(oldSigma))
        error = float(np.max(np.concatenate([
            np.abs(newbeta - oldbeta), np.abs(newf - oldf), np.abs(diag_diff),
        ])))
        iteration += 1
        oldbeta, oldf, oldSigma = newbeta, newf, newSigma
        oldalpha = newf[-1]
    runtime = _time.time() - t0

    estpar = np.concatenate([
        newbeta[1:],
        np.atleast_1d(np.ravel(newSigma)),
        newf * float(np.exp(newbeta[0])),
    ])
    cov = covest(
        ID, Y, XX, ZZ, Delta, n, newf, newbeta, newSigma,
        bb, weight, indY, indID, rho, r1, r2,
    )
    MX = np.zeros((nbeta + ngamma + npt, nbeta + ngamma + nn))
    MX[:nbeta + ngamma, :nbeta + ngamma] = np.eye(nbeta + ngamma)
    MX[nbeta + ngamma:, nbeta + ngamma:] = (testpt[:, None] >= OY[None, :]).astype(float)
    sd = np.diag(MX @ cov @ MX.T)

    if sd.min() > 0:
        beta_hat = MX @ estpar
        se_beta = np.sqrt(sd)
        out_path = os.path.join(
            res_dir, f'npmle_N{n}_seed{seed}_setting_{data_setting}.npz',
        )
        np.savez_compressed(
            out_path,
            beta_hat=beta_hat.reshape(-1, 1),
            se_beta=se_beta.reshape(-1, 1),
            runtime=runtime,
        )
        print('NPMLE finished!')
    else:
        print('Covariance non-positive; result not saved.')


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    n = int(args[0]) if args else 200
    seed = int(args[1]) if len(args) > 1 else 1
    main(n, seed)
