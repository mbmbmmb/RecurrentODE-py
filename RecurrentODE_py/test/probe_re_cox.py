"""Single-seed probe for random_effect.cox:
  1. Fit with main(ci=True) — closed-form SE from inference_beta.
  2. Call inference() (resampling B=50) on the same fit.
  3. Compare the two SE vectors for beta.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time as _time

import numpy as np

from RecurrentODE_py.random_effect.cox.main import main as re_cox_main
from RecurrentODE_py.random_effect.cox.inference import inference as re_cox_resample


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    setting = 1
    root = tempfile.mkdtemp(prefix='re_cox_probe_')
    print(f'workdir: {root}')
    print(f'N={N}, seed={seed}, setting={setting}')

    t0 = _time.time()
    est_r, rt = re_cox_main(N, seed, setting, ci=True, root=root)
    t_main = _time.time() - t0
    print(f'\nmain() closed-form done in {t_main:.2f}s (fit {rt:.2f}s)')

    se_file = os.path.join(
        root, 'res', f'res_cox_N{N}_seed{seed}_setting{setting}_se.npz',
    )
    se_closed = np.load(se_file)['se_beta'].ravel()
    beta_hat = est_r[:3]
    print(f'  beta_hat      = {beta_hat.round(4)}')
    print(f'  SE (closed)   = {se_closed.round(4)}')

    t0 = _time.time()
    fish = re_cox_resample(N, seed, setting, root=root)
    t_inf = _time.time() - t0
    se_resamp_all = np.sqrt(np.abs(np.diag(fish)))
    se_resamp_beta = se_resamp_all[:3]
    print(f'\ninference() resampling (B=50) done in {t_inf:.2f}s')
    print(f'  SE (resamp)   = {se_resamp_beta.round(4)}')
    print(f'  ratio (closed/resamp) = {(se_closed / se_resamp_beta).round(3)}')

    resamp_file = os.path.join(
        root, 'res',
        f'res_cox_N{N}_seed{seed}_setting{setting}_inference.npz',
    )
    print(f'\nfiles written:\n  {se_file}\n  {resamp_file}')


if __name__ == '__main__':
    main()
