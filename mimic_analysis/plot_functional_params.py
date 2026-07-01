"""Plot Cox and LTM functional parameters (log10 scenario).

  - Cox:  log baseline hazard  log lambda_0(t) = B(t) . theta
          (point estimate; sandwich variance is for beta, not theta)
  - LTM:  time-transform alpha(t)         = exp(B_0(t) . alpha)
          and log baseline hazard B_q(t) . theta_LTM
          with pointwise 95% bands from the resampled Fisher matrix.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = '/Users/bomeng/Desktop/research/review/jmlr/code/mimic_analysis'
DATA = os.path.join(ROOT, 'merged_data')
OUT  = os.path.join(ROOT, 'doc/plots/functional')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RecurrentODE_py.common import spcol  # noqa: E402


def grid_for(knots, n=400, eps=1e-6):
    return np.linspace(knots[0], knots[-1] - eps, n)


def pointwise_band(B, mu, cov):
    """Return mean +/- 1.96*SE for each row of B (t-points), where each
    pointwise value is B[i] @ mu and Var = B[i] @ cov @ B[i]."""
    val = B @ mu
    var = np.einsum('ij,jk,ik->i', B, cov, B)
    se = np.sqrt(np.clip(var, 0.0, None))
    return val, val - 1.96 * se, val + 1.96 * se


def plot_cox_baseline(label):
    d = np.load(os.path.join(DATA, f'cox_spline_{label}.npz'),
                allow_pickle=True)
    theta = np.asarray(d['theta']).ravel()
    knots = np.asarray(d['knots']).ravel()
    k = int(np.asarray(d['k']))
    t = grid_for(knots)
    B = spcol(knots, k, t)
    log_h0 = B @ theta

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, np.exp(log_h0), color='C0', lw=2,
            label=r'Cox $\lambda_0(t)=\exp(B(t)\theta)$')
    ax.set_yscale('log')
    ax.set_xlabel(f'time ({"days" if label == "raw" else "log10(1+days)"})')
    ax.set_ylabel(r'baseline hazard $\lambda_0(t)$ (log scale)')
    ax.set_title(f'Cox PH baseline hazard (random-effect, scenario={label})')
    ax.grid(alpha=0.3, which='both')
    ax.legend()
    note = ('Note: Cox sandwich SE is for beta only;\n'
            'theta-variance not saved -> point estimate only.')
    ax.text(0.99, 0.02, note, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    fig.tight_layout()
    p = os.path.join(OUT, f'cox_baseline_{label}.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_ltm_alpha(label):
    """LTM time-transform alpha(t) = exp(B_0(t) . alpha)."""
    d = np.load(os.path.join(DATA, f'ltm_spline_{label}.npz'),
                allow_pickle=True)
    alpha = np.asarray(d['alpha']).ravel()
    knots_0 = np.asarray(d['knots_0']).ravel()
    k0 = int(np.asarray(d['k0']))
    fish = np.asarray(d['fish'])
    p_beta_free = int(np.asarray(d['beta']).size) - 1
    q_q = int(np.asarray(d['theta']).size)
    cov_alpha = fish[p_beta_free + q_q:, p_beta_free + q_q:]

    t = grid_for(knots_0)
    B = spcol(knots_0, k0, t)
    log_a, lo, hi = pointwise_band(B, alpha, cov_alpha)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, np.exp(log_a), color='C2', lw=2,
            label=r'LTM $\alpha(t)=\exp(B_0(t)\alpha)$')
    ax.fill_between(t, np.exp(lo), np.exp(hi),
                    color='C2', alpha=0.2,
                    label='95% pointwise band')
    ax.set_yscale('log')
    ax.axhline(1.0, color='black', ls=':', lw=0.6, alpha=0.5)
    ax.axvline(1.5, color='black', ls=':', lw=0.6, alpha=0.5)
    ax.text(1.5, ax.get_ylim()[0], r'  $\alpha(1.5)=1$',
            ha='left', va='bottom', fontsize=9, color='black')
    ax.set_xlabel(f'time ({"days" if label == "raw" else "log10(1+days)"})')
    ax.set_ylabel(r'$\alpha(t)$ (log scale)')
    ax.set_title(
        f'LTM time-transform integrand alpha(t) (scenario={label})')
    ax.grid(alpha=0.3, which='both')
    ax.legend(loc='best')
    fig.tight_layout()
    p = os.path.join(OUT, f'ltm_alpha_{label}.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_ltm_lambda(label):
    """LTM baseline hazard lambda_0(u) on the transformed time scale."""
    d = np.load(os.path.join(DATA, f'ltm_spline_{label}.npz'),
                allow_pickle=True)
    theta = np.asarray(d['theta']).ravel()
    knots_q = np.asarray(d['knots_q']).ravel()
    kq = int(np.asarray(d['kq']))
    fish = np.asarray(d['fish'])
    p_beta_free = int(np.asarray(d['beta']).size) - 1
    q_q = int(np.asarray(d['theta']).size)
    cov_theta = fish[p_beta_free:p_beta_free + q_q,
                     p_beta_free:p_beta_free + q_q]

    u = grid_for(knots_q)
    B = spcol(knots_q, kq, u)
    log_h, lo, hi = pointwise_band(B, theta, cov_theta)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(u, np.exp(log_h), color='C3', lw=2,
            label=r'LTM $\lambda_0(u)=\exp(B_q(u)\theta)$')
    ax.fill_between(u, np.exp(lo), np.exp(hi),
                    color='C3', alpha=0.2,
                    label='95% pointwise band')
    ax.set_yscale('log')
    ax.set_xlabel('transformed time u')
    ax.set_ylabel(r'baseline hazard $\lambda_0(u)$ (log scale)')
    ax.set_title(
        f'LTM baseline hazard on transformed scale (scenario={label})')
    ax.grid(alpha=0.3, which='both')
    ax.legend()
    fig.tight_layout()
    p = os.path.join(OUT, f'ltm_lambda_{label}.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_combined(label):
    """Side-by-side overview: Cox lambda_0(t), LTM alpha(t), LTM lambda_0(u)."""
    cox = np.load(os.path.join(DATA, f'cox_spline_{label}.npz'),
                  allow_pickle=True)
    ltm = np.load(os.path.join(DATA, f'ltm_spline_{label}.npz'),
                  allow_pickle=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Cox baseline
    theta_c = np.asarray(cox['theta']).ravel()
    knots_c = np.asarray(cox['knots']).ravel()
    kc = int(np.asarray(cox['k']))
    tc = grid_for(knots_c)
    Bc = spcol(knots_c, kc, tc)
    axes[0].plot(tc, np.exp(Bc @ theta_c), color='C0', lw=2)
    axes[0].set_yscale('log')
    axes[0].set_title(r'Cox PH: $\lambda_0(t)$')
    axes[0].set_xlabel(f't ({"days" if label == "raw" else "log10(1+d)"})')
    axes[0].grid(alpha=0.3, which='both')
    axes[0].text(0.02, 0.98, '(point estimate)',
                 transform=axes[0].transAxes, ha='left', va='top',
                 fontsize=9, alpha=0.7)

    # LTM alpha
    p_bf = int(np.asarray(ltm['beta']).size) - 1
    q_q = int(np.asarray(ltm['theta']).size)
    fish = np.asarray(ltm['fish'])
    cov_a = fish[p_bf + q_q:, p_bf + q_q:]
    cov_t = fish[p_bf:p_bf + q_q, p_bf:p_bf + q_q]
    alpha = np.asarray(ltm['alpha']).ravel()
    knots_0 = np.asarray(ltm['knots_0']).ravel()
    k0 = int(np.asarray(ltm['k0']))
    ta = grid_for(knots_0)
    Ba = spcol(knots_0, k0, ta)
    log_a, la, ha = pointwise_band(Ba, alpha, cov_a)
    axes[1].plot(ta, np.exp(log_a), color='C2', lw=2)
    axes[1].fill_between(ta, np.exp(la), np.exp(ha), color='C2', alpha=0.2)
    axes[1].axhline(1.0, color='k', ls=':', lw=0.6, alpha=0.4)
    axes[1].axvline(1.5, color='k', ls=':', lw=0.6, alpha=0.4)
    axes[1].set_yscale('log')
    axes[1].set_title(r'LTM: $\alpha(t)$ (time-transform integrand)')
    axes[1].set_xlabel(f't ({"days" if label == "raw" else "log10(1+d)"})')
    axes[1].grid(alpha=0.3, which='both')

    # LTM lambda
    theta_l = np.asarray(ltm['theta']).ravel()
    knots_q = np.asarray(ltm['knots_q']).ravel()
    kq = int(np.asarray(ltm['kq']))
    u = grid_for(knots_q)
    Bq = spcol(knots_q, kq, u)
    log_h, lh, hh = pointwise_band(Bq, theta_l, cov_t)
    axes[2].plot(u, np.exp(log_h), color='C3', lw=2)
    axes[2].fill_between(u, np.exp(lh), np.exp(hh), color='C3', alpha=0.2)
    axes[2].set_yscale('log')
    axes[2].set_title(r'LTM: $\lambda_0(u)$ (transformed time)')
    axes[2].set_xlabel('u')
    axes[2].grid(alpha=0.3, which='both')

    fig.suptitle(
        f'Functional parameters, scenario={label}', fontsize=12,
        fontweight='bold', y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, f'functional_combined_{label}.png')
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return p


def main():
    written = []
    for label in ['raw', 'log10']:
        try:
            written.append(plot_cox_baseline(label))
        except Exception as e:
            print(f'cox {label}: {e}')
        try:
            written.append(plot_ltm_alpha(label))
        except Exception as e:
            print(f'ltm alpha {label}: {e}')
        try:
            written.append(plot_ltm_lambda(label))
        except Exception as e:
            print(f'ltm lambda {label}: {e}')
        try:
            written.append(plot_combined(label))
        except Exception as e:
            print(f'combined {label}: {e}')
    for p in written:
        print(f'wrote {p}')


if __name__ == '__main__':
    main()
