# Recurrent Event Analysis with Ordinary Differential Equations (Python)

This GitHub repository provides the **Python** implementation of the estimation and inference procedures proposed in the paper:

[Recurrent Event Analysis with Ordinary Differential Equation](https://arxiv.org/abs/2507.20396)

[Bo Meng](https://www.linkedin.com/in/bomeng202/), [Weijing Tang](https://sites.google.com/andrew.cmu.edu/weijingtang/home), [Gongjun Xu](https://sites.google.com/umich.edu/gongjunxu/home), and [Ji Zhu](http://dept.stat.lsa.umich.edu/~jizhu/).

This is a Python port of the original MATLAB code.

## Getting Started

### Prerequisites

The implementation requires Python 3.9+ and the following packages:

* [`numpy`](https://numpy.org/) for numerical arrays and linear algebra
* [`scipy`](https://scipy.org/) for optimization (`scipy.optimize.minimize`) and ODE integration (`scipy.integrate.solve_ivp`)
* [`scipy.interpolate`](https://docs.scipy.org/doc/scipy/reference/interpolate.html) for B-spline construction (sieve space)

Install with:

```bash
pip install numpy scipy
```

### Optional Packages

- [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) (Python standard library) for distributing replications across CPU cores in [test/test_coverage.py](test/test_coverage.py).
- [`matplotlib`](https://matplotlib.org/) for the optional plotting utilities in `*/visual.py` and [test/plot_bands.py](test/plot_bands.py).

## Usage

We first introduce the usage of the proposed estimation method for a Poisson process without random effects.

### The Cox-type Model

Fitting the **Cox-type** recurrent event model:

$$ \mu'_x(t) = \exp(x^\top\beta)\alpha(t)$$

with $q(\cdot)\equiv 1$.

- **Data Generation:** Run [cox/generator_rec.py](cox/generator_rec.py) and set the `data_setting` argument to `1`. The generated data will be saved in the `cox/data/` directory.
- **Proposed ODE-Cox Estimation:** Run [cox/main.py](cox/main.py). This will approximate the functional parameter $\alpha(\cdot)$ using a quadratic B-spline. Moreover, if the `calculate_ci` parameter is set to `True`, it will calculate standard error estimates by inverting the empirical Fisher Information matrix. See [cox/summary.py](cox/summary.py) for example.

### The AFT Model

Fitting the **accelerated failure time (AFT)-type** recurrent event model:

$$ \mu'_x(t) = q(\mu(t))\exp(x^\top\beta)$$

with $\alpha(\cdot)\equiv 1$.

- **Data Generation:** Run [aft/generator_rec.py](aft/generator_rec.py) and set the `data_setting` argument to `2`. The generated data will be saved in the `aft/data/` directory.
- **Proposed ODE-AFT Estimation:** Run [aft/main.py](aft/main.py). This will approximate the functional parameter $q(\cdot)$ using a cubic B-spline. Moreover, if the `ci` parameter is set to `True`, it will calculate standard error estimates by inverting the empirical Fisher Information matrix. See [aft/summary.py](aft/summary.py) for example.

### The G-Transformation Model

Fitting the **semi-parametric linear transformation** recurrent event model:

$$\mu'_x(t) = q(\mu(t))\exp(x^\top\beta)\alpha(t)$$

with a specified transformation function $q(\cdot|\rho)$ and an unspecified functional parameter $\alpha(\cdot)$. The solution of this ODE with the initial value $\mu_x(0) = 0$ corresponds to the G-transformation function in [Zeng and Lin (2007)](https://doi.org/10.1111/j.1369-7412.2007.00606.x), which covers semi-parametric transformation models with Box-Cox and logarithmic transformations.

- **Data Generation:** Run [npmle/generator_rec.py](npmle/generator_rec.py) and set the `data_setting` argument to `3`. The generated data will be saved in the `npmle/data/` directory.
- **Proposed ODE-LT Estimation:** Run [npmle/main.py](npmle/main.py). This will approximate the functional parameter $\alpha(\cdot)$ using a quadratic B-spline. Moreover, if the `ci` parameter is set to `True`, it will calculate standard error estimates by inverting the empirical Fisher Information matrix. See [npmle/summary.py](npmle/summary.py) for example.

### The Linear Transformation Model

Fitting the **general (nonparametric) linear transformation** model:

$$ \mu'_x(t) = q(\mu(t))\exp(x^\top\beta)\alpha(t)$$

with both $q(\cdot)$ and $\alpha(\cdot)$ unspecified.

- **Data Generation:** Run [ltm/generator_rec.py](ltm/generator_rec.py) and set the `data_setting` argument to `4`. The generated data will be saved in the `ltm/data/` directory.
- **Proposed ODE-Flex Estimation:** Run [ltm/main.py](ltm/main.py). This will approximate the functional parameters using B-splines. Moreover, if the `ci` parameter is set to `True`, it will calculate standard error estimates by inverting the empirical Fisher Information matrix. See [ltm/summary.py](ltm/summary.py) for example.

We compare the performance of this ODE-Flex method with existing methods under settings 1) - 4) in the manuscript.

### The Gamma Frailty Model

Fitting the linear transformation model with Gamma-distributed random effects. Specifically, for subject $i$, $\xi_i$ follows a mean-1 Gamma distribution, and the rate function satisfies:

$$ \mu'_{x_i}(t) = \xi_i q(\mu(t))\exp(x_i^\top\beta)\alpha(t)$$

The following settings are available for data generation:

- **Cox-type model:** $\alpha(\cdot)$ is unspecified and $q(\cdot)\equiv 1$.
  - Run [random_effect/cox/generator_rec.py](random_effect/cox/generator_rec.py) and set the `data_setting` argument to `1`.

- **AFT-type model:** $\alpha(\cdot)\equiv 1$ and $q(\cdot)$ unspecified.
  - Run [random_effect/aft_rec/generator_rec.py](random_effect/aft_rec/generator_rec.py) and set the `data_setting` argument to `2`.

The following estimation methods are available:

- **ODE-Cox:** For Cox-type data; has a closed-form solution for the covariance matrix of $\hat{\beta}$.
  - Run [random_effect/cox/main.py](random_effect/cox/main.py).
  - See [random_effect/cox/summary.py](random_effect/cox/summary.py) for example.

- **ODE-AFT:** For AFT-type data.
  - Run [random_effect/aft_rec/main.py](random_effect/aft_rec/main.py).
  - See [random_effect/aft_rec/summary.py](random_effect/aft_rec/summary.py) for example.

- **ODE-Flex:** For both Cox- and AFT-type data.
  - Run [random_effect/ltm/main.py](random_effect/ltm/main.py).
  - See [random_effect/ltm/summary_cox.py](random_effect/ltm/summary_cox.py) and [random_effect/ltm/summary_aft.py](random_effect/ltm/summary_aft.py) for example.

For all three models, the covariance matrix for the regression and spline coefficients can be obtained via the resampling method proposed in [Zeng and Lin (2008)](https://academic.oup.com/biostatistics/article-abstract/9/2/355/354461?redirectedFrom=PDF).

### Informative-Censoring Stress Test

The [inform_censor/](inform_censor/) directory provides a coverage stress test under
**covariate-dependent censoring**: the censoring time $C$ is generated
from the same covariates $x$ as the recurrent-event process but with an
independent random draw, so $C$ and $N(\cdot)$ are conditionally
independent given $x$ and only marginally correlated through $x$.

Six variants are included, crossing two recurrent-event intensities (Cox / AFT) with three censoring families:

| Module | Recurrent events | Censoring distribution |
|---|---|---|
| `inform_censor/cox`        | Cox  | Cox-survival (exponential) |
| `inform_censor/aft`        | AFT  | Cox-survival (exponential) |
| `inform_censor/cox_aftc`   | Cox  | AFT-survival              |
| `inform_censor/aft_aftc`   | AFT  | AFT-survival              |
| `inform_censor/cox_unif`   | Cox  | Uniform on $[a(x), b(x)]$ |
| `inform_censor/aft_unif`   | AFT  | Uniform on $[a(x), b(x)]$ |

See [inform_censor/RESULTS.md](inform_censor/RESULTS.md) for the full coverage table and censoring-rate summary.

### Knot Placement

We implemented two types of knot placement for the polynomial splines. The `knots_setting` argument in `main.py` can be set to:

- `"quantile"`: The interior knots are placed at the quantiles of the distinct event observation times.
- `"equal"`: The interior knots are placed evenly across the time interval from 0 to the maximum observed event time.

We use these two knot placement strategies across various simulation settings.

### Reproducing the Coverage Studies

The [test/test_coverage.py](test/test_coverage.py) script runs `--rep` independent replications in parallel and reports bias, empirical SE, mean sandwich SE, and 95% Wald-CI coverage:

```bash
# Cox-type recurrent events, 100 replications, 10 worker processes
python3 -m RecurrentODE_py.test.test_coverage \
    --module cox --setting 1 --N 1000 --rep 100 --workers 10

# AFT-type recurrent events with equal-spaced knots
python3 -m RecurrentODE_py.test.test_coverage \
    --module aft --setting 2 --knots equal --N 1000 --rep 100 --workers 10
```

The full set of pre-configured runs (including the random-effect and informative-censoring variants) is in [job_submission/configs.json](job_submission/configs.json) and can be launched in batch via [run_local.py](run_local.py).

## Contact

Bo Meng (University of Michigan) - bomeng@umich.edu
